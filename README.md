# metameric

Metameric is a simulator for Interactive Activation (IA) networks.
Interactive Activation networks are localist connectionist models, which means that their neurons uniquely identify a single concept, e.g. a letter for each letter, or a word neuron for each word.
Interactive Activation was first introduced in [McClelland & Rumelhart (1981)](https://www.cs.indiana.edu/~port/teach/641/McClellandRumelhart.IAC.model.1981.pdf), but has been used widely in the field computational psycholinguistics.

Unlike distributed connectionist models, which have been supported by a variety of useful tools and toolkits, no such toolkit exists for localist connectionist modeling.
Metameric intends to fill this gap.

It is first and foremost meant to be a out-of-the-box simulator for the canonical IA model, but is easily extensible to other models, such as TRACE.

# innovations

Metameric includes several innovations which allows the IA model to simulate stimuli of different lengths out of the box, and is, to our knowledge, the first simulator to do so.

These innovations are:

* Negative input features
* Space padding the input
* Weight adaptation

These three innovations are fully explained in the companion paper, which is currently under review.

# Name

The name Metameric comes from the short story ``The White Death`` by polish author [Stanisław Lem](https://en.wikipedia.org/wiki/Stanis%C5%82aw_Lem):

```
The hereditary and at the same time perpetual ruler was Metameric, for he possessed a cold, beautiful and many-membered frame, and in the first of these members resided his mind; when that grew old, after thousands of years, when the crystal networks had been worn away from much administrative thinking, its authority was taken over by the next member, and thus it went, for of these he had ten billion.
```

# Usage

First make sure all requirements in requirements.txt are installed.

### pip
```
pip install -r requirements.txt
```

### conda
```
conda install --yes --file requirements.txt
```

Then install using

```
python3 setup.py install
```

Then, you can run metameric with.

```
python3 -m metameric -i MY_INPUT_FILE -o MY_OUTPUT_FILE
```

Note that is most likely necessary to first move to another folder before running this due to cython dependencies.

For a quick example, use `example.csv` as `MY_INPUT_FILE`
You can also try normal preparation by running the `prepare` function.

```
python3 -m metameric.prepare -i example_orth.csv -o example.csv -d orthography --decomposable_names letters -f letters --feature_sets fourteen
```

This turns a normal word csv with fields for orthography and frequency into data which can be fed into a full IA model.
This can be used with the English Lexicon Project out of the box, for example.

```
python3 -m metameric.prepare -i elp-items.csv -o test.csv -d Word --decomposable_names letters -f letters --feature_sets fourteen --disable_strict
```

`disable_strict` is added because not all items in the elp are completely alpha-numeric, and hence can't be featurized by our feature set

You can also use the web interface.

```
python3 -m metameric.web
```

Again, use `example.csv` as a quick example.
