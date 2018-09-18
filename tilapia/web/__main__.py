"""The page where params are made."""
import os
from bottle import run, route, request, template, static_file
from tilapia.run import make_run, get_model


global m
global max_cycles


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
                  weight=True,
                  space=True,
                  rla=rla_layers,
                  rla_layers=rla_layers,
                  input_layers=inputlayers.split(),
                  output_layers=outputlayers.split(),
                  global_rla=float(rla),
                  step_size=float(step),
                  decay_rate=float(decay),
                  minimum_activation=float(min_val))

    return template("templates/analysis_2.tpl", inputs=inputlayers.split())


@route("/analysis_2", method="POST")
def post_word():

    global m
    global max_cycles
    word = {x: request.forms.get(x).split() for x in m.inputs}
    print(word)
    res = m.activate(word, max_cycles=max_cycles)


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
    ia = request.forms.get("ia")
    step = request.forms.get("step")
    decay = request.forms.get("decay")
    min_val = request.forms.get("min")
    max_cyc = request.forms.get("max")
    threshold = request.forms.get("threshold")
    inputlayers = request.forms.get("inputlayers")
    outputlayers = request.forms.get("outputlayers")
    rla_layers = request.forms.get("rlalayers")

    out_f = "run_{}.csv".format(os.path.splitext(input_file.filename)[0])

    if not param_file:
        weights = None
    else:
        weights = param_file
    words = input_file.file
    test_words = test_file.file

    make_run(words,
             test_words,
             "test.csv",
             weights,
             weight=True,
             space=True,
             threshold=float(threshold),
             rla=rla_layers,
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

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    run(host='localhost', port=8080)
