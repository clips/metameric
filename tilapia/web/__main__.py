"""The page where params are made."""
import os
from bottle import run, route, request, template, static_file
from tilapia.run import make_run, get_model
from tilapia.plot import result_plot
from tilapia.prepare.data import process_and_write
from itertools import tee
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


@route('/static/scripts/<filename:re:.*\.js>')
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
                      f_set.split())

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
                  global_rla=float(rla),
                  step_size=float(step),
                  decay_rate=float(decay),
                  minimum_activation=float(min_val))

    return template("templates/analysis_2.tpl", inputs=sorted(m.inputs))


@route("/analysis_2", method="POST")
def post_word():

    global m
    global max_cycles
    word = {x: request.forms.get(x).split() for x in m.inputs}
    res = m.activate(word, max_cycles=max_cycles)
    f = result_plot(res)
    f.savefig(os.path.join("content", "plot.png"))
    return template("templates/analysis_2.tpl", inputs=sorted(m.inputs))


@route("/experiment", method='POST')
def main_experiment():
    """
    """
    print(request.body.read())
    input_file = request.files.get("path_train")
    param_file = request.files.get("path_param")
    test_file = request.files.get("path_test")
    rla = request.forms.get("rla")
    step = request.forms.get("step")
    decay = request.forms.get("decay")
    min_val = request.forms.get("min")
    max_cyc = request.forms.get("max")
    threshold = request.forms.get("threshold")
    outputlayers = request.forms.get("outputlayers")
    rla_layers = request.forms.get("rlalayers")
    rla_variable = request.forms.get("rlavars")

    out_f = "run_{}.csv".format(os.path.splitext(input_file.filename)[0])

    if not param_file:
        weights = None
    else:
        weights = param_file
    if not test_file:
        words, test_words = tee(input_file.file, 2)
    else:
        words = input_file.file
        test_words = test_file.file

    global junk

    make_run(words,
             test_words,
             os.path.join(junk, "test.csv"),
             weights,
             threshold=float(threshold),
             rla_variable=rla_variable,
             rla_layers=rla_layers,
             output_layers=outputlayers.split(),
             global_rla=float(rla),
             step_size=float(step),
             max_cycles=int(max_cyc),
             decay_rate=float(decay),
             minimum_activation=float(min_val))

    return static_file("test.csv", root=junk, download=out_f)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-j",
                        "--junk",
                        type=str,
                        help="Path to an optional junk folder.")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    global junk
    if args.junk:
        junk = args.junk
    else:
        junk = os.getcwd()

    run(host='localhost', port=8080)
