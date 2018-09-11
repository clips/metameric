"""The page where params are made."""
from bottle import run, route, request, template
from tilapia.run import make_run


@route("/", method='GET')
def default():
    return template("tilapia/web/template.tpl")


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
    input_path = request.forms.get("path")
    param_path = request.forms.get("path_param")
    output_path = request.forms.get("output_path")
    rla = request.forms.get("rla")
    step = request.forms.get("step")
    decay = request.forms.get("decay")
    min_val = request.forms.get("min")
    max_cyc = request.forms.get("max")
    threshold = request.forms.get("threshold")
    inputlayers = request.forms.get("inputlayers")
    outputlayers = request.forms.get("outputlayers")
    rla_layers = request.forms.get("rlalayers")

    if param_path == "":
        param_path = None
    make_run(input_path,
             output_path,
             param_path,
             True,
             True,
             threshold,
             rla_layers,
             rla_layers,
             inputlayers,
             outputlayers,
             rla,
             step,
             max_cyc,
             decay,
             min_val)

    return template("tilapia/web/loading.tpl")


if __name__ == "__main__":

    run(host='localhost', port=8080, reloader=False)
