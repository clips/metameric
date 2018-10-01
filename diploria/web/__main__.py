"""The page where params are made."""
import os
import io
import base64

from flask import Flask, render_template as template, request, Response
from diploria.run import make_run, get_model
from diploria.plot import result_plot
from diploria.prepare.data import process_and_write
from itertools import chain
from argparse import ArgumentParser


global m
global max_cycles
global files
global plot

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')


@app.route("/about", methods=['GET'])
def about():
    return template("about.tpl")


@app.route("/prepare", methods=['GET'])
def prepare():
    return template("prepare.tpl")


@app.route("/prepare", methods=['POST'])
def prepare_post():
    input_file = request.files.get("path_train")
    d_layer = request.form.get("decomp_layer")
    d_name = request.form.get("decomp_name")
    f_layers = request.form.get("feature_layer")
    f_set = request.form.get("feature_set")

    out_f = "result_{}".format(os.path.splitext(input_file.filename)[0])
    junk_file = io.StringIO()
    process_and_write(input_file,
                      junk_file,
                      d_layer.split(),
                      d_name.split(),
                      f_layers.split(),
                      f_set.split(),
                      strict=False)

    junk_file = junk_file.getvalue()
    return Response(response=junk_file,
                    mimetype="text/csv",
                    headers={"Content-disposition":
                             "attachment; filename={}".format(out_f)})


@app.route("/home", methods=['GET'])
def home():
    return template("home.tpl")


@app.route("/", methods=['GET'])
def default():
    return template("home.tpl")


@app.route("/experiment", methods=['GET'])
def experiment():
    return template("experiment.tpl")


@app.route("/analysis", methods=['GET'])
def get_analysis():
    return template("analysis.tpl")


@app.route("/analysis", methods=['POST'])
def analysis():
    """Analyze an IA model."""
    input_file = request.files.get("path_train")
    param_file = request.files.get("path_param")
    rla = request.form.get("rla")
    step = request.form.get("step")
    decay = request.form.get("decay")
    min_val = request.form.get("min")
    max_cyc = request.form.get("max")
    outputlayers = request.form.get("outputlayers")
    rla_layers = request.form.get("rlalayers")
    rla_variable = request.form.get("rlavars")
    w = request.form.get("w")
    monitorlayers = request.form.get("monitorlayers")

    if not param_file:
        weights = None
    else:
        weights = param_file
    words = input_file

    global m
    global max_cycles
    max_cycles = int(max_cyc)
    m = get_model(words,
                  weights,
                  rla_variable=rla_variable,
                  rla_layers=rla_layers,
                  output_layers=outputlayers.split(),
                  monitor_layers=monitorlayers.split(),
                  global_rla=float(rla),
                  step_size=float(step),
                  decay_rate=float(decay),
                  minimum_activation=float(min_val),
                  adapt_weights=bool(w))

    inputs = [[l.name for l in x.to_connections] for x in m.inputs.values()]
    inputs = sorted(set(chain.from_iterable(inputs)))

    return template("analysis_2.tpl", inputs=inputs, data="")


@app.route("/analysis_2", methods=["POST"])
def post_item():
    """Post an item and show the graph."""
    global m
    global max_cycles

    inputs = [[l.name for l in x.to_connections] for x in m.inputs.values()]
    inputs = sorted(set(chain.from_iterable(inputs)))

    item = {}
    for x in inputs:
        max_length = max([y for x, y in m[x].name2idx.keys()])
        max_length += 1
        data = request.form.get(x).ljust(max_length)
        data = data[:max_length]
        item[x] = [(data[idx], idx)
                   for idx in range(max_length)]

    item = m.expand(item)
    res = m.activate([item], max_cycles=max_cycles, strict=False)[0]
    f = result_plot(item, res, {k: m[k].node_names for k in res.keys()})
    image = io.BytesIO()
    f.canvas.print_png(image)
    img = base64.b64encode(image.getvalue())
    img = str(img)[2:-1]
    return template("analysis_2.tpl", inputs=inputs, data=img)


@app.route("/experiment", methods=['POST'])
def main_experiment():
    """The main experiment page."""
    input_file = request.files.get("path_train")
    param_file = request.files.get("path_param")
    test_file = request.files.get("path_test")
    rla = request.form.get("rla")
    step = request.form.get("step")
    decay = request.form.get("decay")
    min_val = request.form.get("min")
    max_cyc = request.form.get("max")
    threshold = request.form.get("threshold")
    rla_layers = request.form.get("rlalayers")
    rla_variable = request.form.get("rlavars")
    w = request.form.get("w")
    monitorlayers = request.form.get("monitorlayers")

    out_f = "run_{}.csv".format(os.path.splitext(input_file.filename)[0])

    if not param_file:
        weights = None

    input_file = input_file
    test_file = test_file

    junk_file = io.StringIO()

    make_run(input_file,
             test_file,
             junk_file,
             weights,
             threshold=float(threshold),
             rla_variable=rla_variable,
             rla_layers=rla_layers,
             output_layers=monitorlayers.split(),
             monitor_layers=monitorlayers.split(),
             global_rla=float(rla),
             step_size=float(step),
             max_cycles=int(max_cyc),
             decay_rate=float(decay),
             minimum_activation=float(min_val),
             adapt_weights=bool(w))

    junk_file = junk_file.getvalue()
    return Response(response=junk_file,
                    mimetype="text/csv",
                    headers={"Content-disposition":
                             "attachment; filename={}".format(out_f)})


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-p",
                        "--port",
                        default=8080,
                        type=int,
                        help="The port on which to run.")
    parser.add_argument("--host",
                        default="localhost",
                        type=str,
                        help="The host to use.")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    print("Running with {} as local directory.".format(os.getcwd()))
    app.run(host=args.host, port=args.port)
