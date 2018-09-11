<!DOCTYPE html>
<html>
<head></head>
<body>
<form action="/" method="post">
    Path to CSV: <input name="path" type="text" /><br>
    Path to Parameter file: <input name="path_param" type="text" /><br>
    Output file: <input name="path_output" type="text" /><br>
    Resting level Activation: <input name="rla" type="text" value="-.05"><br>
    Step size: <input name="step" type="text" value="1.0"><br>
    Decay: <input name="decay" type="text" value=".07"><br>
    Minimum Activation: <input name="min" type="text" value="-.02"><br>
    Maximum Cycles: <input name="min" type="text" value="350"><br>
    Threshold: <input name="threshold" type="text" value=".7"><br>
    Input layers: <input name="inputlayers" type="text" value="features features_neg"/><br>
    Output layers: <input name="outputlayers" type="text" value="orthography"/><br>
    RLA layers: <input name="rlalayers" type="text" value="orthography" /><br>
    <button type="submit" class="btn btn-default">Submit</button>
</form>
</body>
</html>
