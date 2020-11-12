python3 train.py --model_type=GCN --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500
python3 train.py --model_type=GraphSage --hidden_dim=256 --dataset=cora --dropout=0.6 --weight_decay=5e-4 --epochs=600
python3 train.py --model_type=GAT --hidden_dim=64 --dataset=cora --dropout=0.6 --weight_decay=5e-4 --epochs=600
python3 train.py --model_type=APPNP --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500
python train.py --dataset=enzymes --weight_decay=5e-3 --num_layers=3 --epochs=500

