#!/bin/sh

# All the graphs for running the two versions of HEVC/H.265
# with the neural networks for intra prediction are frozen.
python freezing_graph_pnn.py 4 0 1.0 0.0 , --is_fully_connected
python freezing_graph_pnn.py 8 0 1.0 0.0 , --is_fully_connected
python freezing_graph_pnn.py 16 0 1.0 0.0 ,
python freezing_graph_pnn.py 32 0 1.0 0.0 ,
python freezing_graph_pnn.py 64 0 1.0 0.0 ,

# `is_pair` means that the PNN models were trained on
# contexts with HEVC quantization noise.
python freezing_graph_pnn.py 4 0 1.0 0.0 , --is_fully_connected --is_pair
python freezing_graph_pnn.py 8 0 1.0 0.0 , --is_fully_connected --is_pair
python freezing_graph_pnn.py 16 0 1.0 0.0 , --is_pair
python freezing_graph_pnn.py 32 0 1.0 0.0 , --is_pair
python freezing_graph_pnn.py 64 0 1.0 0.0 , --is_pair


