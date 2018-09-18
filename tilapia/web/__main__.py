"""The page where params are made."""
import os
from bottle import run, route, request, template, static_file
from tilapia.run import make_run, get_model
from tilapia.plot import result_plot
from argparse import ArgumentParser


global m
global max_cycles
global junk_folder


@route('/content/<filename:re:.*\.css>')
def stylesheets(filename):
    return static_file(filename, root='content/')


@route('/content/<filename:re:.*>')
def pictures(filename):
    return static_file(filename, root='content/')


@route('/js/<filename:re:.*\.js>')
def javascript(filename):
    return static_file(filename, root='js/')


@route("/contact", method='GET')
def contact():
    return template("templates/contact.tpl")


@route("/about", method='GET')
def about():
    return template("templates/about.tpl")


@route("/home", method='GET')
def home():
    return template("templates/home.tpl")


@route("/", method='GET')
def default():
    return template("templates/home.tpl")


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
    inputlayers = request.forms.get("inputlayers")
    outputlayers = request.forms.get("outputlayers")
    rla_layers = request.forms.get("rlalayers")
    rla_variable = request.forms.get("rlavars")

    if not param_file:
        weights = None
    else:
        weights = param_file
    words = input_file.file
    # test_words = test_file.file

    global m
    global max_cycles
    max_cycles = int(max_cyc)
    m = get_model(words,
                  weights,
                  rla_variable=rla_variable,
                  rla_layers=rla_layers,
                  input_layers=inputlayers.split(),
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


@route("/", method='POST')
def show_plot():
    """
    Path to CSV: <input name="path" type="text" />
    Path to Parameter file: <input name="path_param" type="text" />
    Output file: <input name="path_output" type="text" />
    Resting level Activation: <input name="rla" type="text" />
    Step size: <input name="step" type="text" />
    Decay: <input name="decay" type="text" />
    Minimum Activation: <input name="min" type="text" />
    Maximum Cycles: <input name="max" type="text" />
    Threshold: <input name="threshold" type="text" />
    Input layers: <input name="inputlayers" type="text" />
    Output layers: <input name="outputlayers" type="text" />
    RLA layers: <input name="rlalayers" type="text" />
    """
    input_file = request.files.get("path_train")
    param_file = request.files.get("path_param")
    test_file = request.files.get("path_test")
    rla = request.forms.get("rla")
    step = request.forms.get("step")
    decay = request.forms.get("decay")
    min_val = request.forms.get("min")
    max_cyc = request.forms.get("max")
    threshold = request.forms.get("threshold")
    inputlayers = request.forms.get("inputlayers")
    outputlayers = request.forms.get("outputlayers")
    rla_layers = request.forms.get("rlalayers")
    rla_variable = request.forms.get("rlavars")

    out_f = "run_{}.csv".format(os.path.splitext(input_file.filename)[0])

    if not param_file:
        weights = None
    else:
        weights = param_file
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
             input_layers=inputlayers.split(),
             output_layers=outputlayers.split(),
             global_rla=float(rla),
             step_size=float(step),
             max_cycles=int(max_cyc),
             decay_rate=float(decay),
             minimum_activation=float(min_val))

    return static_file("test.csv", root='.', download=out_f)


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
