# E-ReaRev
In this paper, we introduce a new few-shot setting, which considers the inadequacy of training data and the incompleteness of knowledge graphs in real-world applications. Furthermore, we propose an efficient model for question answering over incomplete knowledge graphs. 

# Run
python main.py

# Folders
new_data: few-shot data derived from 3 orginial datasets (WebQSP, CWQ, MetaQA-3)

modules: basically the same as ReaRev

models: abandon_encoder.py denotes the new encoder used in meaning-extension module, layer_init.py denotes the type layer and the relation-extension layer in edge-extension module
