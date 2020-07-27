manhattan_model
-----

# ref codes
- [cnn model](https://github.com/Shawn1993/cnn-text-classification-pytorch)
- [mahattan lstm](https://github.com/fionn-mac/Manhattan-LSTM)


# environments

```
python3
torch==1.5.1
```

# install

```
$ git clone study_sentence_embedding

$ cd study_sentence_embedding

$ pip install -r requirements.txt

$ wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

# run

## lstm

```
$ time CUDA_VISIBLE_DEVICES=0 python main.py --data_name sick --data_file dataset/SICK.tsv --num_iters 10 --learning_rate 0.001 --model_name=lstm
```

## cnn

```
$ time CUDA_VISIBLE_DEVICES=0 python main.py --data_name sick --data_file dataset/SICK.tsv --num_iters 10 --learning_rate 0.001 --model_name=cnn
```

# result

## lstm

```
0m 55s (- 0m 0s) (10 100%) 89.3750
Validation Accuracy: 86.331301 Validation Precision: 86.331301 Validation Recall: 100.000000 Validation Loss: 50.246347


seq1      > ['two', 'dogs', 'lawn', 'playing', 'plastic', 'toy']
seq2      > ['two', 'dogs', 'running', 'catching', 'tennis', 'ball']
ref score = 3.799999952316284
/data1/users/daniellee/pytorch/Manhattan-LSTM/PyTorch/.venv/lib/python3.6/site-packages/torch/nn/modules/loss.py:432: UserWarning: Using a target size (torch.Size([])) that is different to the input size (torch.Size([1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
cand score = 1.9795472621917725
seq1      > ['person', 'folding', 'piece', 'paper']
seq2      > ['person', 'folding', 'piece', 'paper']
ref score = 3.5
cand score = 4.511545181274414
seq1      > ['surfer', 'falling', 'wave']
seq2      > ['surfer', 'riding', 'wave']
ref score = 3.799999952316284
cand score = 3.6867709159851074
seq1      > ['men', 'talking']
seq2      > ['old', 'women', 'talking']
ref score = 3.0
cand score = 4.004637241363525
seq1      > ['person', 'motocross', 'uniform', 'wearing', 'helmet', 'rides', 'red', 'motorcycle']
seq2      > ['person', 'motocross', 'uniform', 'wearing', 'helmet', 'riding', 'red', 'motorcycle']
ref score = 4.0
cand score = 3.988440990447998
seq1      > ['man', 'passionately', 'playing', 'guitar', 'front', 'audience']
seq2      > ['woman', 'peeling', 'potato']
ref score = 1.0
cand score = 1.1279287338256836
seq1      > ['man', 'white', 'hat', 'touching', 'cardboard', 'box']
seq2      > ['man', 'black', 'hat', 'reaching', 'box']
ref score = 4.199999809265137
cand score = 1.9438308477401733
seq1      > ['woman', 'wearing', 'ear', 'protection', 'firing', 'gun', 'outdoor', 'shooting', 'range']
seq2      > ['woman', 'shooting', 'target', 'practice']
ref score = 4.099999904632568
cand score = 3.9691553115844727
seq1      > ['little', 'dog', 'dropping', 'bedroom', 'slipper', 'mouth']
seq2      > ['little', 'dog', 'grabbing', 'bedroom', 'slipper', 'mouth']
ref score = 3.5999999046325684
cand score = 4.070366859436035
seq1      > ['golden', 'dog', 'running', 'field', 'tall', 'grass']
seq2      > ['golden', 'dog', 'running', 'field', 'tall', 'grass']
ref score = 4.900000095367432
cand score = 3.8498127460479736
```

## cnn

```
Validation Accuracy: 86.229675 Validation Precision: 86.229675 Validation Recall: 100.000000 Validation Loss: 69.640228


seq1      > ['man', 'clumsily', 'kick', 'boxing', 'trainer']
seq2      > ['karate', 'practitioner', 'kicking', 'another', 'man', 'wearing', 'protective', 'boxing', 'gloves']
ref score = 2.799999952316284
/data1/users/daniellee/pytorch/Manhattan-LSTM/PyTorch/.venv/lib/python3.6/site-packages/torch/nn/modules/loss.py:432: UserWarning: Using a target size (torch.Size([])) that is different to the input size (torch.Size([1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
cand score = 2.805457353591919
seq1      > ['man', 'elegant', 'dress', 'surrounded', 'photographers']
seq2      > ['man', 'surrounded', 'photographers', 'wearing', 'gray', 'suit', 'glasses']
ref score = 3.299999952316284
cand score = 2.90108060836792
seq1      > ['someone', 'giving', 'food', 'animal']
seq2      > ['somebody', 'playing', 'piano']
ref score = 1.2999999523162842
cand score = 2.154412031173706
seq1      > ['two', 'people', 'race', 'flipping', 'tires', 'tractor']
seq2      > ['two', 'women', 'competing', 'tire', 'rolling', 'race']
ref score = 3.799999952316284
cand score = 2.571408748626709
seq1      > ['girl', 'riding', 'horse']
seq2      > ['man', 'riding', 'horse']
ref score = 2.9000000953674316
cand score = 3.3375253677368164
seq1      > ['man', 'sitting', 'train', 'resting', 'hand', 'face']
seq2      > ['man', 'standing', 'train', 'resting', 'hand', 'lap']
ref score = 3.4000000953674316
cand score = 3.7290406227111816
seq1      > ['jockeys', 'racing', 'horses', 'field', 'completely', 'green']
seq2      > ['jockeys', 'slowing', 'horses', 'field', 'completely', 'green']
ref score = 4.199999809265137
cand score = 3.7318477630615234
seq1      > ['two', 'bmx', 'bikers', 'jumping', 'dirt', 'ramps', 'front', 'body', 'water']
seq2      > ['bikers', 'acrobatics', 'track', 'near', 'water', 'skyline']
ref score = 3.299999952316284
cand score = 3.529340982437134
seq1      > ['military', 'officer', 'shouting', 'recruits']
seq2      > ['military', 'officer', 'barking', 'recruits']
ref score = 4.800000190734863
cand score = 4.710611820220947
seq1      > ['man', 'dancing']
seq2      > ['man', 'motionless']
ref score = 3.299999952316284
cand score = 2.5985827445983887
```
