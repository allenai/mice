# Local imports
from src.stage_one import run_train_editor 
from src.utils import get_args, load_predictor, get_dataset_reader

if __name__ == '__main__':

    args = get_args("stage1")
    predictor = load_predictor(args.meta.task)
    dr = get_dataset_reader(args.meta.task, predictor)
    run_train_editor(predictor, dr, args)
