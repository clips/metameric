"""The page where params are made."""
import os
from bottle import run, route, request, template, static_file
from tilapia.run import make_run, get_model
from tilapia.plot import result_plot
from tilapia.prepare.data import process_and_write
from itertools import chain
from argparse import ArgumentParser


global m
global max_cycles
global junk_folder


@route('/static/content/<filename:re:.*\.css>')
def stylesheets(filename):
    return static_file(filename, root='static/content')


@route('/static/content/<filename:re:.*>')
def pictures(filename):
    return static_file(filename, root='static/content')


@route('/static/scripts/<filename:re:.*>')
def javascript(filename):
    return static_file(filename, root='static/scripts/')


@route("/contact", method='GET')
def contact():
    return template("templates/contact.tpl")


@route("/about", method='GET')
def about():
    return template("templates/about.tpl")


@route("/prepare", method='GET')
def prepare():
    return template("templates/prepare.tpl")


@route("/prepare", method='POST')
def prepare_post():
    input_file = request.files.get("path_train")
    d_layer = request.forms.get("decomp_layer")
    d_name = request.forms.get("decomp_name")
    f_layers = request.forms.get("feature_layer")
    f_set = request.forms.get("feature_set")

    out_f = "prepared-{}".format(input_file.filename)

    process_and_write(input_file.file,
                      os.path.join(junk, "data.csv"),
                      d_layer.split(),
                      d_name.split(),
                      f_layers.split(),
                      f_set.split(),
                      strict=False)

    return static_file("data.csv", root=junk, download=out_f)


@route("/home", method='GET')
def home():
    return template("templates/home.tpl")


@route("/", method='GET')
def default():
    return template("templates/home.tpl")


@route("/experiment", method='GET')
def experiment():
    return template("templates/experiment.tpl")


@route("/analysis", method='GET')
def get_analysis():
    return template("templates/analysis.tpl")


@route("/analysis", method='POST')
def analysis():
    """Analyze an IA model."""
    input_file = request.files.get("path_train")
    param_file = request.files.get("path_param")
    rla = request.forms.get("rla")
    step = request.forms.get("step")
    decay = request.forms.get("decay")
    min_val = request.forms.get("min")
    max_cyc = request.forms.get("max")
    outputlayers = request.forms.get("outputlayers")
    rla_layers = request.forms.get("rlalayers")
    rla_variable = request.forms.get("rlavars")
    w = request.forms.get("w")
    monitorlayers = request.forms.get("monitorlayers")

    if not param_file:
        weights = None
    else:
        weights = param_file
    words = input_file.file

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

    return template("templates/analysis_2.tpl", inputs=inputs)


@route("/analysis_2", method="POST")
def post_word():
    """Post a word and show the graph."""
    global m
    global max_cycles

    inputs = [[l.name for l in x.to_connections] for x in m.inputs.values()]
    inputs = sorted(set(chain.from_iterable(inputs)))

    word = {}
    for x in inputs:
        max_length = max([y for x, y in m[x].name2idx.keys()])
        max_length += 1
        data = request.forms.get(x).ljust(max_length)
        data = data[:max_length]
        word[x] = [(data[idx], idx)
                   for idx in range(max_length)]

    word = m.expand(word)
    res = m.activate([word], max_cycles=max_cycles, strict=False)[0]
    f = result_plot(word, res, {k: m[k].node_names for k in res.keys()})
    f.savefig(os.path.join("static/content", "plot.png"))

    return template("templates/analysis_2.tpl", inputs=inputs)


@route("/experiment", method='POST')
def main_experiment():
    """The main experiment page."""
    input_file = request.files.get("path_train")
    param_file = request.files.get("path_param")
    test_file = request.files.get("path_test")
    rla = request.forms.get("rla")
    step = request.forms.get("step")
    decay = request.forms.get("decay")
    min_val = request.forms.get("min")
    max_cyc = request.forms.get("max")
    threshold = request.forms.get("threshold")
    rla_layers = request.forms.get("rlalayers")
    rla_variable = request.forms.get("rlavars")
    w = request.forms.get("w")
    monitorlayers = request.forms.get("monitorlayers")

    out_f = "run_{}.csv".format(os.path.splitext(input_file.filename)[0])

    if not param_file:
        weights = None

    input_file = input_file.file
    test_file = test_file.file

    global junk

    make_run(input_file,
             test_file,
             os.path.join(junk, "test.csv"),
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

    return static_file("test.csv", root=junk, download=out_f)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-j",
                        "--junk",
                        type=str,
                        help="Path to an optional junk folder.")
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

    global junk
    if args.junk:
        junk = args.junk
    else:
        junk = os.getcwd()
    try:
        os.remove(os.path.join(junk, "static/content/plot.png"))
    except FileNotFoundError:
        pass

    print("Running with {} as local directory.".format(os.getcwd()))

    run(host=args.host, port=args.port)
