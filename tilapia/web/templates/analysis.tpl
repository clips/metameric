%rebase('templates/base.tpl', title='Index')
<form action="/analysis" method="post" enctype="multipart/form-data">
    <div class="form-group row mb-0">
        <label for="path_train" class="col-sm-5 col-form-label">Path to training file</label>
        <div class="col-sm-5">
            <input type="file" class="form-control form-control-sm" id="path_train" name="path_train" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="path_param" class="col-sm-5 col-form-label">Path to parameter file</label>
        <div class="col-sm-5">
            <input type="file" class="form-control form-control-sm" id="path_param" name="path_param">
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="rla" class="col-sm-5 col-form-label">RLA</label>
        <div class="col-sm-5">
            <input type="number" class="form-control form-control-sm" id="rla" value="-0.05" step="0.01" min="-1.0" max="1.0" name="rla" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="step" class="col-sm-5 col-form-label">Step size</label>
        <div class="col-sm-5">
            <input type="number" class="form-control form-control-sm" id="step" value="1.0" step="0.1" min=".1" max="1.0" name="step" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="decay" class="col-sm-5 col-form-label">Decay</label>
        <div class="col-sm-5">
            <input type="number" class="form-control form-control-sm" id="decay" value="0.07" step="0.01" min="0.01" max="1.0" name="decay" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="min" class="col-sm-5 col-form-label">Minimum activation</label>
        <div class="col-sm-5">
            <input type="number" class="form-control form-control-sm" id="min" value="-0.02" min="-1.0" step=".01" max="1.0" name="min" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="max" class="col-sm-5 col-form-label">Maximum cycles</label>
        <div class="col-sm-5">
            <input type="number" class="form-control form-control-sm" id="max" value="350" min="1" max="100000" name="max" value="350" min="1" max="100000" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="threshold" class="col-sm-5 col-form-label">Threshold</label>
        <div class="col-sm-5">
            <input type="number" class="form-control form-control-sm" id="threshold" value=".7" name="threshold" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="outputlayers" class="col-sm-5 col-form-label">Output layers</label>
        <div class="col-sm-5">
            <input type="text" class="form-control form-control-sm" id="outputlayers" value="orthography" name="outputlayers" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="monitorlayers" class="col-sm-5 col-form-label">Monitor Layers</label>
        <div class="col-sm-5">
            <input type="text" class="form-control form-control-sm" id="monitorlayers" value="orthography" name="monitorlayers" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="rlalayers" class="col-sm-5 col-form-label">RLA layers</label>
        <div class="col-sm-5">
            <input type="text" class="form-control form-control-sm" id="rlalayers" value="orthography" name="rlalayers" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="rlavars" class="col-sm-5 col-form-label">RLA variable</label>
        <div class="col-sm-5">
            <input type="text" class="form-control form-control-sm" id="rlavars" value="frequency" name="rlavars" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="w" class="col-sm-5 col-form-label">Weight adaptation</label>
        <div class="col-sm-5">
            <input type="checkbox" class="form-control form-control-sm" id="w" value="w" checked name="w">
        </div>
    </div>
    <button type="submit" class="btn btn-default btn-sm">Submit</button>
</form>
