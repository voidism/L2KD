A=woz.en
B=cnn_dailymail
C=wikisql
SEED=$1

bash train.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $A $B $C > log.train.kdlll.ABC.$SEED 2>&1
sleep 30
bash test.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $A $B $C > log.test.kdlll.ABC.$SEED 2>&1
sleep 30
bash train.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $A $C $B > log.train.kdlll.ACB.$SEED 2>&1
sleep 30
bash test.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $A $C $B > log.test.kdlll.ACB.$SEED 2>&1
sleep 30
bash train.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $B $C $A > log.train.kdlll.BCA.$SEED 2>&1
sleep 30
bash test.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $B $C $A > log.test.kdlll.BCA.$SEED 2>&1
sleep 30
bash train.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $B $A $C > log.train.kdlll.BAC.$SEED 2>&1
sleep 30
bash test.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $B $A $C > log.test.kdlll.BAC.$SEED 2>&1
sleep 30
bash train.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $C $A $B > log.train.kdlll.CAB.$SEED 2>&1
sleep 30
bash test.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $C $A $B > log.test.kdlll.CAB.$SEED 2>&1
sleep 30
bash train.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $C $B $A > log.train.kdlll.CBA.$SEED 2>&1
sleep 30
bash test.sh --seq_train_type lll --model_name gpt2 --add_task_tokens --seed $SEED --gen_lm_sample_percentage 0.2 --distil --seq_distil --tasks $C $B $A > log.test.kdlll.CBA.$SEED 2>&1

