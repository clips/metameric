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
Note that is most likely necessary to first move to another folder before running this due to cython dependencies.

```
python3 -m tilapia -i MY_INPUT_FILE -o MY_OUTPUT_FILE
```

For a quick example, use `example.csv` as `MY_INPUT_FILE`

You can also try normal preparation by running the `prepare` function.

```
python3 -m tilapia.prepare -i example_orth.csv -o example.csv -d orthography --decomposable_names letters -f letters --feature_names fourteen
```

This turns a normal word csv with fields for orthography and frequency into data which can be fed into a full IA model.


You can also use the web interface.

```
python3 -m tilapia.web
```

Again, use `example.csv` as a quick example.
In the web demo you will sometimes encounter a thing called `Parameter File`.
This should be left empty.
