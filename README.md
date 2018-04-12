# Post-Specialisation
Post-Specialisation: Retrofitting Vectors of Words Unseen in Lexical Resources (Vulić et al., NAACL-HLT 2018)

This repository contains the code and data for the post-specialisation method in the NAACL-HLT 2018 paper. 

Contact: Ivan Vulić (iv250@cam.ac.uk)

### Configuring the Tool

The post-specialisation tool reads all the experiment config parameters from the ```experiment_parameters.cfg``` file in the root directory. An alternative config file can be provided as the first (and only) argument to ```code/post-specialisation.py```. 

Note that the tool assumes that input distributional vectors have already been initially specialised with one of the standard specialisation models such as [retrofitting](https://github.com/mfaruqui/retrofitting) (Faruqui et al., NAACL-HLT 2015), [counter-fitting](https://github.com/nmrksic/counter-fitting) (Mrkšić et al., NAACL-HLT 2016), or [Attract-Repel](https://github.com/nmrksic/attract-repel) (Mrkšić et al., TACL 2017). In our experiments, we use the most recent and state-of-the-art post-processor Attract-Repel, but the post-specialisation model is equally applicable to any other post-processor. If you use any of these tools, please cite the corresponding paper(s). 

The config file specifies the following:
1. The location of the initial word vectors (```distributional_vectors```)
- In the default setup, we use SGNS vectors with bag-of-words contexts trained on Wikipedia, available [here](http://u.cs.biu.ac.il/~yogo/data/syntemb/bow2.words.bz2)
* the location of the training data; training data contains word vectors (x_i, x_o) changed by the initial specialisation (i.e., seen words x)
2. We have to specify the location of the distributional vectors (x_i) in ```distributional_training_data``` as well as the location of the specialised vectors (x_o) in ```specialised_training_data```
- The two training data files follow the standard format for word vectors (word dim_1 dim_2 ... dim_N), but note that they have to be aligned and have exactly the same number of items (i.e., the representation of the same word in both files has to be on the same line).
- We have provided toy sample training data files containing 1000 training pairs to illustrate the data format.
3. The config file also specifies the hyperparameters of the LEAR procedure (set to their default values in ```config/experiment_parameters.cfg```).

### Running Experiments

```python code/post-specialisation.py config/experiment_parameters.cfg```

Running the experiment loads the word vectors specified in the config file and learns the mapping/regression function using a deep feed-forward network as specified in the config file. The procedure prints the updated word vectors to the results directory as ```results/final_vectors.txt``` (standard format: one word vector per line)

### References

The paper which introduces the LEAR procedure:
```
 @inproceedings{Vulic:2018,
  author    = {Ivan Vuli\'{c} and Goran Glava\v{s} and Nikola Mrk\v{s}i\'c} and Anna Korhonen,
  title     = {Post-Specialisation: Retrofitting Vectors of Words Unseen in Lexical Resources},
  booktitle   = {Proceedings of NAACL-HLT},
  year      = 2018,
 }
```
