srun --pty --partition=1080ti-long --gres=gpu:1 --mem=32GB --ntasks-per-node=16 /bin/bash
conda activate event-relation-extraction
cd ~/git/CE2ERE/
export PYTHONPATH="/home/tianyiyang/git/CE2ERE/"

python src/__main__.py --data_dir=data --data_type=hieve --downsample=0.05 --epochs=50 --lambda_anno=1 --lambda_trans=0 --learning_rate=1e-06 --log_batch_size=5 --lstm_hidden_size=256 --lstm_input_size=768 --mlp_size=512 --num_layers=1 --seed=2479266196

python src/__main__.py --data_dir=data --data_type=hieve --downsample=0.15 --epochs=50 --lambda_anno=1 --lambda_trans=0 --learning_rate=1e-06 --log_batch_size=5 --lstm_hidden_size=256 --lstm_input_size=768 --mlp_size=512 --num_layers=1 --seed=2479266196

python src/__main__.py --data_dir=data --data_type=hieve --downsample=0.25 --epochs=50 --lambda_anno=1 --lambda_trans=0 --learning_rate=1e-06 --log_batch_size=5 --lstm_hidden_size=256 --lstm_input_size=768 --mlp_size=512 --num_layers=1 --seed=2479266196

python src/__main__.py --data_dir=data --data_type=hieve --downsample=0.35 --epochs=50 --lambda_anno=1 --lambda_trans=0 --learning_rate=1e-06 --log_batch_size=5 --lstm_hidden_size=256 --lstm_input_size=768 --mlp_size=512 --num_layers=1 --seed=2479266196

python src/__main__.py --data_dir=data --data_type=hieve --downsample=0.45 --epochs=50 --lambda_anno=1 --lambda_trans=0 --learning_rate=1e-06 --log_batch_size=5 --lstm_hidden_size=256 --lstm_input_size=768 --mlp_size=512 --num_layers=1 --seed=2479266196

-------------------------------------------------------------------------

python src/__main__.py --model=box --data_dir=data --data_type=matres --downsample=0.35 --epochs=50 --lambda_anno=1 --lambda_trans=0 --learning_rate=1e-05 --log_batch_size=5 --lstm_hidden_size=256 --lstm_input_size=768 --mlp_size=512 --num_layers=1 --seed=2479266196

python src/__main__.py --model=box --data_dir=data --data_type=hieve --downsample=0.35 --epochs=50 --lambda_anno=1 --lambda_trans=0 --learning_rate=1e-05 --log_batch_size=5 --lstm_hidden_size=256 --lstm_input_size=768 --mlp_size=512 --num_layers=1 --seed=2479266196