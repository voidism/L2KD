A=amazon
B=yelp
C=yahoo
D=ag
E=dbpedia
SEED=$1

bash train.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $A $B $C $D $E > log.train.seqkd.TC.$SEED 2>&1
sleep 30
bash test.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $A $B $C $D $E > log.test.seqkd.TC.$SEED 2>&1
sleep 30
bash train.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $E $D $C $B $A > log.train.seqkd.TC.rev.$SEED 2>&1
sleep 30
bash test.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $E $D $C $B $A > log.test.seqkd.TC.rev.$SEED 2>&1
