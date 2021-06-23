import torch
import numpy as np
import re
import os
import sys
import more_itertools as mit
import math
import textwrap
import logging
import warnings

# Local imports
from src.utils import * 

logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)

class Editor():
    def __init__(
        self, 
        tokenizer_wrapper, 
        tokenizer, 
        editor_model, 
        masker, 
        num_gens = 15, 
        num_beams = 30, 
        grad_pred = "contrast", 
        generate_type = "sample", 
        no_repeat_ngram_size = 2, 
        top_k = 30, 
        top_p = 0.92, 
        length_penalty = 0.5, 
        verbose = True, 
        prepend_label = True,
        ints_to_labels = None
    ):
        self.tokenizer = tokenizer
        self.device = get_device()
        self.num_gens = num_gens
        self.editor_model = editor_model.to(self.device)
        self.tokenizer_wrapper = tokenizer_wrapper
        self.masker = masker
        if ints_to_labels is None:
            ints_to_labels = get_ints_to_labels(self.masker.predictor)
        self.ints_to_labels = ints_to_labels
        self.max_length = self.editor_model.config.n_positions
        self.predictor = self.masker.predictor
        self.dataset_reader = self.predictor._dataset_reader 
        self.grad_pred = grad_pred
        self.verbose = verbose 
        self.generate_type = generate_type
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.top_k = top_k
        self.top_p = top_p
        self.length_penalty = length_penalty 
        self.num_beams = num_beams
        self.prepend_label = prepend_label

    def get_editor_input(self, targ_pred_label, masked_editable_seg, *args):
        """ Format input for editor """

        prefix = "" if not self.prepend_label else "label: " + \
                targ_pred_label + ". input: " 
        return prefix + masked_editable_seg

    def get_editable_seg_from_input(self, inp):
        """ Map whole input -> editable seg. 
        These are the same for single-input classification. """

        return inp

    def get_input_from_editable_seg(self, inp, editable_seg):
        """ Map whole input -> editable seg. 
        These are the same for IMDB/Newsgroups. """
        
        return editable_seg 

    def truncate_editable_segs(self, editable_segs, **kwargs):
        """ Truncate editable segments to max length of Predictor. """ 

        trunc_es = [None] * len(editable_segs)
        for s_idx, s in enumerate(editable_segs):
            assert(len(s) > 0)
            predic_tokenized = get_predictor_tokenized(self.predictor, s)
            
            max_predic_tokens = self.dataset_reader._tokenizer._max_length
            if len(predic_tokenized) >= max_predic_tokens: 
                for idx, token in enumerate(reversed(predic_tokenized)):
                    if token.idx_end is not None:
                        last_idx = token.idx_end
                        break
                trunc_es[s_idx] = s[0:last_idx]
            else:
                trunc_es[s_idx] = s
        return trunc_es

    def input_to_instance(self, inp, editable_seg = None, return_tuple = False):
        """ Convert input to AllenNLP instance object """

        if editable_seg is None:
            instance = self.dataset_reader.text_to_instance(inp)
        else:
            instance = self.dataset_reader.text_to_instance(editable_seg)
        if return_tuple:
            # TODO: hacky bc for race dataset reader, we return length list
            return instance, [None] 
        return instance

    def get_sorted_token_indices(self, inp, grad_pred_idx):
        """ Get token indices to mask, sorted by gradient value """

        editable_seg = self.get_editable_seg_from_input(inp)
        editable_toks = self.tokenizer_wrapper.tokenize(editable_seg)[:-1]
        sorted_token_indices = self.masker.get_important_editor_tokens(
                editable_seg, grad_pred_idx, editable_toks, 
                num_return_toks = len(editable_toks)
                )
        return sorted_token_indices 

    def get_candidates(
            self, targ_pred_label, inp, targ_pred_idx, orig_pred_idx,
            sorted_token_indices = None):
        """ Gets edit candidates after infilling with Editor. 
        Returns dicts with edited inputs (i.e. whole inputs, dicts in the case
        of RACE) and edited editable segs (i.e. just the parts of inputs
        that are editable, articles in the case of RACE). """
       
        assert targ_pred_idx != orig_pred_idx

        if self.grad_pred == "contrast":
            grad_pred_idx = targ_pred_idx 
        elif self.grad_pred == "original":
            grad_pred_idx = orig_pred_idx 
        else:
            raise ValueError

        num_spans, token_ind_to_mask, masked_inp, orig_spans, max_length = \
                self._prepare_input_for_editor(
                        inp, targ_pred_idx, grad_pred_idx,
                        sorted_token_indices=sorted_token_indices)

        edited_editable_segs = self._sample_edits(targ_pred_label, inp, 
                masked_inp, targ_pred_idx, num_spans=num_spans, 
                orig_spans=orig_spans, max_length=max_length)
        edited_cands = [None] * len(edited_editable_segs)
        for idx, es in enumerate(edited_editable_segs):
            cand = {}
            es = self.truncate_editable_segs([es], inp=inp)[0]
            cand['edited_input'] = self.get_input_from_editable_seg(inp, es) 
            cand['edited_editable_seg'] = es 
            edited_cands[idx] = cand 

        return edited_cands, masked_inp

    def _prepare_input_for_editor(
            self, inp, targ_pred_idx, grad_pred_idx,
            sorted_token_indices = None):
        """ Helper function that prepares masked input for Editor. """
        
        tokens = self.tokenizer_wrapper.tokenize(inp)[:-1]
        tokens = [t.text for t in tokens]
       
        if sorted_token_indices is not None: 
            num_return_toks = math.ceil(self.masker.mask_frac * len(tokens))
            token_ind_to_mask = sorted_token_indices[:num_return_toks]
            grouped_ind_to_mask, token_ind_to_mask, masked_inp, orig_spans = \
                    self.masker.get_masked_string(
                            inp, grad_pred_idx, 
                            editor_mask_indices=token_ind_to_mask
                            )

        else:
            grouped_ind_to_mask, token_ind_to_mask, masked_inp, orig_spans = \
                    self.masker.get_masked_string(inp, grad_pred_idx)

        max_length = math.ceil((self.masker.mask_frac + 0.2) * \
                len(sorted_token_indices))
        num_spans = len(grouped_ind_to_mask)

        return num_spans, token_ind_to_mask, masked_inp, orig_spans, max_length

    def _process_gen(self, masked_inp, gen, sentinel_toks):
        """ Helper function that processes decoded gen """

        bad_gen = False
        first_bad_tok = None

        # Hacky: If sentinel tokens are consecutive, then re split won't work
        gen = gen.replace("><extra_id", "> <extra_id")

        # Remove <pad> prefix, etc.
        gen = gen[gen.find("<extra_id_0>"):]

        # Sanity check
        assert not gen.startswith(self.tokenizer.pad_token) 

        # This is for baseline T5 which does not handle masked last tokens well.
        # Baseline often predicts </s> as last token instead of sentinel tok.
        # Heuristically treating the first </s> tok as the final sentinel tok.
        # TODO: will this mess things up for non-baseline editor_models?
        if sentinel_toks[-1] not in gen:
            first_eos_token_idx = gen.find(self.tokenizer.eos_token)
            gen = gen[:first_eos_token_idx] + sentinel_toks[-1] + \
                    gen[first_eos_token_idx + len(self.tokenizer.eos_token):]

        last_sentin_idx = gen.find(sentinel_toks[-1])
        if last_sentin_idx != -1:
        # If the last token we are looking for is in generation, truncate 
            gen = gen[:last_sentin_idx + len(sentinel_toks[-1])]

        # If </s> is in generation, truncate 
        eos_token_idx = gen.find(self.tokenizer.eos_token)
        if eos_token_idx != -1:
            gen = gen[:eos_token_idx]

        # Check if every sentinel token is in the gen
        for x in sentinel_toks:
            if x not in gen:
                bad_gen = True
                first_bad_tok = self.tokenizer.encode(x)[0]
                break
        
        tokens = list(filter(None, re.split(
                            '<extra_id_.>|<extra_id_..>|<extra_id_...>', gen)))
        gen_sentinel_toks = re.findall(
                            '<extra_id_.>|<extra_id_..>|<extra_id_...>', gen)

        gen_sentinel_toks = gen_sentinel_toks[:len(tokens)]

        temp = masked_inp 
        ctr = 0
        prev_temp = temp
        tok_sentinel_iterator = zip(tokens, gen_sentinel_toks)
        for idx, (token, sentinel_tok) in enumerate(tok_sentinel_iterator):
            sentinel_idx = sentinel_tok[-2:-1] if len(sentinel_tok) == 12 \
                    else sentinel_tok[-3:-1]
            sentinel_idx = int(sentinel_idx)
            
            # Check order of generated sentinel tokens 
            if sentinel_idx != ctr:
                first_bad_tok = self.tokenizer.encode(f"<extra_id_{ctr}>")[0]
                bad_gen = True
                break

            if idx != 0:
                temp = temp.replace(prev_sentinel_tok, prev_token)
            prev_sentinel_tok = sentinel_tok
            prev_token = token
            
            # If last replacement, make sure final sentinel token was generated
            is_last = (idx == len(tokens)-1)
            if is_last and gen_sentinel_toks[-1] in sentinel_toks and not bad_gen:
                if " " + sentinel_tok in temp:
                    temp = temp.replace(" " + sentinel_tok, token)
                elif "-" + sentinel_tok in temp:
                    # If span follows "-" character, remove first white space
                    if token[0] == " ":
                        token = token[1:]
                    temp = temp.replace(sentinel_tok, token)
                else:
                    temp = temp.replace(sentinel_tok, token)
            else:
                first_bad_tok = self.tokenizer.encode("<extra_id_{ctr}>")[0]
            ctr += 1

        return bad_gen, first_bad_tok, temp, gen 

    def _get_pred_with_replacement(self, temp_gen, orig_spans, *args):
        """ Replaces sentinel tokens in gen with orig text and returns pred. 
        Used for intermediate bad generations. """

        orig_tokens = list(filter(None, re.split(
                                '<extra_id_.>|<extra_id_..>', orig_spans)))
        orig_sentinel_toks = re.findall('<extra_id_.>|<extra_id_..>', orig_spans)

        for token, sentinel_tok in zip(orig_tokens, orig_sentinel_toks[:-1]):
            if sentinel_tok in temp_gen:
                temp_gen = temp_gen.replace(sentinel_tok, token)
        temp_instance = self.dataset_reader.text_to_instance(temp_gen)
        return temp_gen, self.predictor.predict_instance(temp_instance)
        
        
    def _sample_edits(
            self, targ_pred_label, inp, masked_editable_seg, targ_pred_idx, 
            num_spans = None, orig_spans = None, max_length = None):
        """ Returns self.num_gens copies of masked_editable_seg with infills.
        Called by get_candidates(). """
        
        self.editor_model.eval()       

        editor_input = self.get_editor_input(
                targ_pred_label, masked_editable_seg, inp)
        
        editor_inputs = [editor_input]
        editable_segs = [masked_editable_seg]
        span_end_offsets = [num_spans]
        orig_token_ids_lst = [self.tokenizer.encode(orig_spans)[:-1]]
        orig_spans_lst = [orig_spans]
        masked_token_ids_lst = [self.tokenizer.encode(editor_input)[:-1]]

        k_intermediate = 3 

        sentinel_start = self.tokenizer.encode("<extra_id_99>")[0]
        sentinel_end = self.tokenizer.encode("<extra_id_0>")[0]

        num_sub_rounds = 0
        edited_editable_segs = [] # list of tuples with meta information

        max_sub_rounds = 3
        while editable_segs != []:
       
            # Break if past max sub rounds 
            if num_sub_rounds > max_sub_rounds:
                break
            
            new_editor_inputs = []
            new_editable_segs = []
            new_span_end_offsets = []
            new_orig_token_ids_lst = []
            new_orig_spans_lst = []
            new_masked_token_ids_lst = []
            num_inputs = len(editor_inputs)

            iterator = enumerate(zip(
                editor_inputs, editable_segs, masked_token_ids_lst, 
                span_end_offsets, orig_token_ids_lst, orig_spans_lst))
            for inp_idx, (editor_input, editable_seg, masked_token_ids, \
                    span_end, orig_token_ids, orig_spans) in iterator: 

                num_inputs = len(editor_inputs)
                num_return_seqs = int(math.ceil(self.num_gens/num_inputs)) \
                        if num_sub_rounds != 0 else self.num_gens
                num_beams = self.num_beams if num_sub_rounds == 0 \
                        else num_return_seqs
                last_sentin = f"<extra_id_{span_end}>"
                end_token_id = self.tokenizer.convert_tokens_to_ids(last_sentin)
                masked_token_ids_tensor = torch.LongTensor(
                            masked_token_ids + [self.tokenizer.eos_token_id]
                        ).unsqueeze(0).to(self.device)
                eos_id = self.tokenizer.eos_token_id
                bad_tokens_ids = [[x] for x in range(
                                    sentinel_start, end_token_id)] + [[eos_id]]
                max_length = max(int(4/3 * max_length), 200)
                logger.info(wrap_text("Sub round: " + str(num_sub_rounds)))    
                logger.info(wrap_text(f"Input: {inp_idx} of {num_inputs-1}"))
                logger.info(wrap_text(f"Last sentinel: {last_sentin}"))
                logger.info(wrap_text("INPUT TO EDITOR: " + \
                        f"{self.tokenizer.decode(masked_token_ids)}"))

                with torch.no_grad():
                    if self.generate_type == "beam":
                        output = self.editor_model.generate(
                            input_ids=masked_token_ids_tensor, 
                            num_beams=num_beams, 
                            num_return_sequences=num_return_seqs, 
                            no_repeat_ngram_size=self.no_repeat_ngram_size, 
                            eos_token_id=end_token_id, 
                            early_stopping=True, 
                            length_penalty=self.length_penalty, 
                            bad_words_ids=bad_tokens_ids, 
                            max_length=max_length) 

                    elif self.generate_type == "sample":
                        output = self.editor_model.generate(
                            input_ids=masked_token_ids_tensor, 
                            do_sample=True, 
                            top_p=self.top_p, 
                            top_k=self.top_k, 
                            num_return_sequences=num_return_seqs, 
                            no_repeat_ngram_size=self.no_repeat_ngram_size, 
                            eos_token_id=end_token_id, 
                            early_stopping=True, 
                            length_penalty=self.length_penalty,
                            bad_words_ids=bad_tokens_ids, 
                            max_length=max_length) 
                output = output.cpu()
                del masked_token_ids_tensor 
                torch.cuda.empty_cache()

                batch_decoded = self.tokenizer.batch_decode(output)
                num_gens_with_pad = 0
                num_bad_gens = 0
                temp_edited_editable_segs = []
                logger.info(wrap_text("first batch: " + batch_decoded[0]))
                for batch_idx, batch in enumerate(batch_decoded):
                    sentinel_toks = [f"<extra_id_{idx}>" for idx in \
                            range(0, span_end + 1)]
                    bad_gen, first_bad_tok, temp, stripped_batch = \
                            self._process_gen(editable_seg, batch, sentinel_toks)

                    if len(sentinel_toks) > 3: 
                        assert sentinel_toks[-2] in editor_input

                    if "<pad>" in batch[4:]:
                        num_gens_with_pad += 1
                    if bad_gen:
                        
                        num_bad_gens += 1
                        temp_span_end_offset = first_bad_tok - end_token_id + 1

                        new_editable_token_ids = np.array(
                                self.tokenizer.encode(temp)[:-1])

                        sentinel_indices = np.where(
                                (new_editable_token_ids >= sentinel_start) & \
                                (new_editable_token_ids <= sentinel_end))[0]  
                        new_first_token = max(
                                new_editable_token_ids[sentinel_indices])
                        diff = sentinel_end - new_first_token
                        new_editable_token_ids[sentinel_indices] += diff
                        
                        new_span_end_offsets.append(len(sentinel_indices))

                        new_editable_seg = self.tokenizer.decode(
                                new_editable_token_ids)
                        new_editable_segs.append(new_editable_seg)
                        
                        new_input = self.get_editor_input(targ_pred_label, 
                                new_editable_seg, inp)

                        new_masked_token_ids = self.tokenizer.encode(new_input)[:-1]
                        new_masked_token_ids_lst.append(new_masked_token_ids)

                        # Hacky but re-decode to remove spaces b/w sentinels
                        new_editor_input = self.tokenizer.decode(
                                new_masked_token_ids)
                        new_editor_inputs.append(new_editor_input)

                        # Get orig token ids from new first token and on
                        new_orig_token_ids = np.array(orig_token_ids[np.where(
                            orig_token_ids == new_first_token)[0][0]:]) 
                        sentinel_indices = np.where((
                            new_orig_token_ids >= sentinel_start) & \
                            (new_orig_token_ids <= sentinel_end))[0]
                        new_orig_token_ids[sentinel_indices] += diff
                        new_orig_token_ids_lst.append(new_orig_token_ids)
                        new_orig_spans = self.tokenizer.decode(new_orig_token_ids)
                        new_orig_spans_lst.append(new_orig_spans)

                    else:
                        temp_edited_editable_segs.append(temp)
                        assert "<extra_id" not in temp
                    
                    assert "</s>" not in temp

                edited_editable_segs.extend(temp_edited_editable_segs)

            if new_editor_inputs == []:
                break

            _, unique_batch_indices = np.unique(new_editor_inputs, 
                    return_index=True)

            targ_probs = [-1] * len(new_editable_segs)
            for idx in unique_batch_indices:
                ot = new_orig_spans_lst[idx].replace("<pad>", "")
                temp, pred = self._get_pred_with_replacement(
                        new_editable_segs[idx], ot, inp)
                pred = add_probs(pred)
                targ_probs[idx] = pred['probs'][targ_pred_idx]
                predicted_label = self.ints_to_labels[np.argmax(pred['probs'])] 
                contrast_label = self.ints_to_labels[targ_pred_idx]
                if predicted_label == contrast_label: 
                    edited_editable_segs.append(temp)

            highest_indices = np.argsort(targ_probs)[-k_intermediate:]
            filt_indices = [idx for idx in highest_indices \
                    if targ_probs[idx] != -1]
            editor_inputs = [new_editor_inputs[idx] for idx in filt_indices]
            editable_segs = [new_editable_segs[idx] for idx in filt_indices]
            span_end_offsets = [new_span_end_offsets[idx] for idx in filt_indices] 
            orig_token_ids_lst = [new_orig_token_ids_lst[idx] for idx in filt_indices] 
            orig_spans_lst = [new_orig_spans_lst[idx] for idx in filt_indices] 
            masked_token_ids_lst = [new_masked_token_ids_lst[idx] for idx in filt_indices] 

            sys.stdout.flush()
            num_sub_rounds += 1

        for idx, es in enumerate(edited_editable_segs):
            assert es.find("</s>") in [len(es)-4, -1]
            edited_editable_segs[idx] = es.replace("</s>", " ")
            assert "<extra_id_" not in es, \
                    f"Extra id token should not be in edited inp: {es}"
            assert "</s>" not in es, \
                    f"</s> should not be in edited inp: {edited_editable_segs[idx][0]}"


        return set(edited_editable_segs) 

class RaceEditor(Editor):
    def __init__(
            self, 
            tokenizer_wrapper, 
            tokenizer, 
            editor_model, 
            masker, 
            num_gens = 30, 
            num_beams = 30, 
            grad_pred = "contrast", 
            generate_type = "sample", 
            length_penalty = 1.0, 
            no_repeat_ngram_size = 2, 
            top_k = 50, 
            top_p = 0.92, 
            verbose = False, 
            editable_key = "article"
        ):
        super().__init__(
                tokenizer_wrapper, tokenizer, editor_model, masker, 
                num_gens=num_gens, num_beams=num_beams, 
                ints_to_labels=[str(idx) for idx in range(4)], 
                grad_pred=grad_pred, 
                generate_type=generate_type, 
                no_repeat_ngram_size=no_repeat_ngram_size, 
                top_k=top_k, top_p=top_p, 
                length_penalty=length_penalty, 
                verbose=verbose)
        
        self.editable_key = editable_key
        if self.editable_key not in ["question", "article"]:
            raise ValueError("Invalid value for editable_key")

    def _get_pred_with_replacement(self, temp_gen, orig_spans, inp):
        """ Replaces sentinel tokens in gen with orig text and returns pred. 
        Used for intermediate bad generations. """

        orig_tokens = list(filter(None, re.split(
            '<extra_id_.>|<extra_id_..>|<extra_id_...>', orig_spans)))
        orig_sentinel_toks = re.findall(
                '<extra_id_.>|<extra_id_..>|<extra_id_...>', orig_spans)

        for token, sentinel_tok in zip(orig_tokens, orig_sentinel_toks[:-1]):
            if sentinel_tok in temp_gen:
                temp_gen = temp_gen.replace(sentinel_tok, token)
        # temp_gen is article for RACE
        temp_instance = self.dataset_reader.text_to_instance(
                inp["id"], temp_gen, inp["question"], inp["options"])[0]
        return temp_gen, self.predictor.predict_instance(temp_instance)

    def get_editable_seg_from_input(self, inp):
        """ Map whole input -> editable seg. """ 
        
        return inp[self.editable_key]

    def get_input_from_editable_seg(self, inp, editable_seg):
        """ Map editable seg -> whole input. """ 

        new_inp = inp.copy()
        new_inp[self.editable_key] = editable_seg
        return new_inp

    def truncate_editable_segs(self, editable_segs, inp = None):
        """ Truncate editable segments to max length of Predictor. """ 
        
        trunc_inputs = [None] * len(editable_segs)
        instance, length_lst, max_length_lst = self.input_to_instance(
                inp, return_tuple = True)
        for s_idx, es in enumerate(editable_segs):
            editable_toks = get_predictor_tokenized(self.predictor, es)
            predic_tok_end_idx = len(editable_toks)
            predic_tok_end_idx = min(predic_tok_end_idx, max(max_length_lst))
            last_index = editable_toks[predic_tok_end_idx - 1].idx_end
            editable_seg = es[:last_index]
            trunc_inputs[s_idx] = editable_seg
        return trunc_inputs

    def get_editor_input(self, targ_pred_label, masked_editable_seg, inp):
        """ Format input for editor """
        
        options = inp["options"]
        if masked_editable_seg is None:
            article = inp["article"]
            question = inp["question"]
        else: # masked editable input given
            if self.editable_key == "article":
                article = masked_editable_seg
                question = inp["question"]
            elif self.editable_key == "question":
                article = inp["article"] 
                question = masked_editable_seg 

        editor_input = format_multiple_choice_input(
                article, question, options, int(targ_pred_label))
        return editor_input

    def input_to_instance(
            self, inp, editable_seg = None, return_tuple = False):
        """ Convert input to AllenNLP instance object """
        
        if editable_seg is None:
            article = inp["article"]
            question = inp["question"]
        else: # editable input given
            if self.editable_key == "article":
                article = editable_seg
                question = inp["question"]
            elif self.editable_key == "question":
                article = inp["article"] 
                question = editable_seg
        output = self.dataset_reader.text_to_instance(
                inp["id"], article, question, inp["options"])
        if return_tuple:
            return output
        return output[0]

    def get_sorted_token_indices(self, inp, grad_pred_idx):
        """ Get token indices to mask, sorted by gradient value """

        editable_seg = self.get_editable_seg_from_input(inp)

        inst, length_lst, _ = self.input_to_instance(inp, return_tuple=True)
        editable_toks = get_predictor_tokenized(self.predictor, editable_seg)
        num_editab_toks = len(editable_toks)

        predic_tok_end_idx = len(editable_toks)
        predic_tok_end_idx = min(
                predic_tok_end_idx, length_lst[grad_pred_idx])
        
        if self.editable_key == "article":
            predic_tok_start_idx = 0 
        elif self.editable_key == "question":
            predic_tok_start_idx = length_lst[grad_pred_idx]
            predic_tok_end_idx = length_lst[grad_pred_idx] + num_editab_toks 
        
        editable_toks = self.tokenizer_wrapper.tokenize(editable_seg)[:-1]
        sorted_token_indices = self.masker.get_important_editor_tokens(
                editable_seg, grad_pred_idx, editable_toks,
                num_return_toks=len(editable_toks), 
                labeled_instance=inst, 
                predic_tok_end_idx=predic_tok_end_idx, 
                predic_tok_start_idx=predic_tok_start_idx)
        return sorted_token_indices 
        
    def _prepare_input_for_editor(self, inp, targ_pred_idx, grad_pred_idx,
            sorted_token_indices = None):
        """ Helper function that prepares masked input for Editor. """

        editable_seg = self.get_editable_seg_from_input(inp)

        tokens = [t.text for t in \
                self.tokenizer_wrapper.tokenize(editable_seg)[:-1]]

        instance, length_lst, _ = self.input_to_instance(
                inp, return_tuple=True)
        editable_toks = get_predictor_tokenized(self.predictor, editable_seg) 
        num_editab_toks = len(editable_toks)
        predic_tok_end_idx = len(editable_toks)
        predic_tok_end_idx = min(
                predic_tok_end_idx, length_lst[grad_pred_idx])
        
        if self.editable_key == "article":
            predic_tok_start_idx = 0 
        elif self.editable_key == "question":
            predic_tok_start_idx = length_lst[grad_pred_idx]
            predic_tok_end_idx = length_lst[grad_pred_idx] + num_editab_toks

        if sorted_token_indices is not None: 
            num_return_toks = math.ceil(
                    self.masker.mask_frac * len(tokens))
            token_ind_to_mask = sorted_token_indices[:num_return_toks]

            grouped_ind_to_mask, token_ind_to_mask, masked_inp, orig_spans = \
                    self.masker.get_masked_string(editable_seg, grad_pred_idx, 
                            editor_mask_indices=token_ind_to_mask, 
                            predic_tok_start_idx=predic_tok_start_idx, 
                            predic_tok_end_idx=predic_tok_end_idx)

        else:
            grouped_ind_to_mask, token_ind_to_mask, masked_inp, orig_spans = \
                    self.masker.get_masked_string(
                            editable_seg, grad_pred_idx, 
                            labeled_instance=instance, 
                            predic_tok_end_idx=predic_tok_end_idx, 
                            predic_tok_start_idx=predic_tok_start_idx)

        num_spans = len(grouped_ind_to_mask)
        max_length = math.ceil(
                (self.masker.mask_frac+0.2) * len(sorted_token_indices))

        masked_inp = masked_inp.replace(self.tokenizer.eos_token, " ")
        return num_spans, token_ind_to_mask, masked_inp, orig_spans, max_length
