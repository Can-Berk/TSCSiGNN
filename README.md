# TSCSiGNN

`train_signn.py` important hyperparameters of SiGNN are as follows:
*   `--dataset` dataset name
*   `--gnn`: Graph Neural Network operator
*   `--sim_measure`: DTW or attention-based approach for graph construction and edge weights
*   `--lr`: the learning rate set for training
*   `--alpha`: scaling factor of dis(similarity) weights to build an adjacency matrix
*   `--dilated`: use dilated convolutions in ResNet feature extractor (default=True)
*   `--epochs`: the number of epochs for model training
*   `--K`: the number of closest neighbors from a dis(similarity) matrix
*   `--supervision`: switch between supervised and semi-supervised settings transductive/inductive (default=supervised)

For example:
```
python train_signn.py --dataset Handwriting --gnn GATv2Conv --sim_measure dtw --lr 5e-4 --alpha 1 --dilated --epochs 1500
```
