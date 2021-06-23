for TASK in imdb newsgroups race
do	
	mkdir -p trained_predictors/${TASK}/model
	mkdir -p results/${TASK}/editors/mice/
	wget https://storage.googleapis.com/allennlp-public-models/mice-${TASK}-predictor.tar.gz -O trained_predictors/${TASK}/model/model.tar.gz
	wget https://storage.googleapis.com/allennlp-public-models/mice-${TASK}-editor.pth -O results/${TASK}/editors/mice/${TASK}_editor.pth 
done

