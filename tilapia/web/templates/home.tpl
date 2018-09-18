%rebase('templates/base.tpl', title='Index')
<form action="/" method="post" enctype="multipart/form-data">
    <div class="grid">
        <label for="path_train">Path to training file</label>
        <input id=atrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="file" id="path_train" name="path_train">
        <label for="path_test">Path to test CSV</label>
        <input id=btrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="file" id="path_test" name="path_test">
        <label for="ia">Use IA structure?</label>
        <input id=btrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="checkbox" id="ia" name="ia">
        <label for="path_param">Path to parameter file</label>
        <input id=ctrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="file" id="path_param" name="path_param">
        <label for="rla">RLA</label>
        <input id=dtrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="text" id="rla" name="rla" value=-.05>
        <label for="step">Step size</label>
        <input id=etrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="text" id="step" name="step" value=1.0>
        <label for="decay">Decay rate</label>
        <input id=ftrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="text" id="decay" name="decay" value=.07>
        <label for="min">Minimum activation</label>
        <input id=gtrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="text" id="min" name="min" value="-.2">
        <label for="max" id="max">Maximum Cycles</label>
        <input id=htrain type=button value="?" class="btn btn-default btn-sm">
        <input type="text" id="max" name="max" value=350>
        <label for="threshold" id="threshold">Threshold</label>
        <input id=itrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="text" id="threshold" name="threshold" value=.7>
        <label for="inputlayers" id="inputlayers">Input layers</label>
        <input id=jtrain type=button value="?" class="btn btn-default btn-sm"/>
        <input name="inputlayers" type="text" value="features features_neg"/>
        <label for="outputlayers" id="outputlayers">Output layers</label>
        <input id=ktrain type=button value="?" class="btn btn-default btn-sm"/>
        <input name="outputlayers" type="text" value="orthography"/>
        <label for="rlalayers" id="rlalayers">RLA layers</label>
        <input id=ltrain type=button value="?" class="btn btn-default btn-sm"/>
        <input name="rlalayers" type="text" value="orthography" />
        <label for="rlavars" id="rlavars">RLA variable</label>
        <input id=ltrain type=button value="?" class="btn btn-default btn-sm"/>
        <input name="rlavars" type="text" value="frequency" />

        </div>
    <button type="submit" class="btn btn-default btn-sm">Submit</button>
</form>
