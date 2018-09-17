"""The page where params are made."""
import os
from bottle import run, route, request, template, static_file
from tilapia.run import make_run


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
    input_path = request.files.get("path_train").filename
    param_path = request.files.get("path_param")
    test_path = request.files.get("path_test").filename
    rla = request.forms.get("rla")
    step = request.forms.get("step")
    decay = request.forms.get("decay")
    min_val = request.forms.get("min")
    max_cyc = request.forms.get("max")
    threshold = request.forms.get("threshold")
    inputlayers = request.forms.get("inputlayers")
    outputlayers = request.forms.get("outputlayers")
    rla_layers = request.forms.get("rlalayers")

    out_f = "run_{}.csv".format(os.path.splitext(input_path)[0])
    if not param_path:
        param_path = None
    else:
        param_path = param_path.filename
    make_run(input_file=input_path,
             test_file=test_path,
             output_file=out_f,
             parameter_file=param_path,
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
             minimum_activation=float(min_val),
             check_output=False)

    return static_file(out_f, root='.', download=out_f)


if __name__ == "__main__":

    run(host='localhost', port=8080, reloader=True, debug=True)
