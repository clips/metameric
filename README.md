# tilapia

First make sure all requirements in requirements.txt are installed.

```
pip install -r requirements.txt
```

Then install using

```
python3 setup.py install
```

Then, you can run tilapia with.

```
python3 -m tilapia -i MY_INPUT_FILE -o MY_OUTPUT_FILE
```

Note that is most likely necessary to first move to another folder before running this due to cython dependencies.

For a quick example, use `example.csv` as `MY_INPUT_FILE`

You can also try normal preparation by running the `prepare` function.

```
python3 -m tilapia.prepare -i example_orth.csv -o example.csv -d orthography --decomposable_names letters -f letters --feature_sets fourteen
```

This turns a normal word csv with fields for orthography and frequency into data which can be fed into a full IA model.
This can be used with e.g. the English Lexicon Project out of the box.

```
python3 -m tilapia.prepare -i elp-items.csv -o test.csv -d Word --decomposable_names letters -f letters --feature_sets fourteen --disable_strict
```

`disable_strict` is added because not all items in the elp are completely alpha-numeric, and hence can't be featurized by our feature set

You can also use the web interface.

```
python3 -m tilapia.web
```

Again, use `example.csv` as a quick example.
In the web demo you will sometimes encounter a thing called `Parameter File`.
This should be left empty.
