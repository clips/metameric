%rebase('templates/base.tpl', title='Index')
<form action="/prepare" method="post" enctype="multipart/form-data">
    <div class="form-group row mb-0">
        <label for="path_train" class="col-sm-5 col-form-label">Path to training file</label>
        <div class="col-sm-5">
            <input type="file" class="form-control form-control-sm" id="path_train" name="path_train" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="decomp_layer" class="col-sm-5 col-form-label">Decomposable Layers</label>
        <div class="col-sm-5">
            <input type="text" value="orthography" class="form-control form-control-sm" id="decomp_layer" name="decomp_layer" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="decomp_name" class="col-sm-5 col-form-label">Names of decomposable Layers</label>
        <div class="col-sm-5">
            <input type="text" value="letters" class="form-control form-control-sm" id="decomp_layer" name="decomp_name" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="decomp_name" class="col-sm-5 col-form-label">Feature layers</label>
        <div class="col-sm-5">
            <input type="text" value="letters" class="form-control form-control-sm" id="feature_layer" name="feature_layer" required>
        </div>
    </div>
    <div class="form-group row mb-0">
        <label for="decomp_name" class="col-sm-5 col-form-label">Feature sets</label>
        <div class="col-sm-5">
            <input type="text" value="fourteen" class="form-control form-control-sm" id="feature_set" name="feature_set" required>
        </div>
    </div>
    <button type="submit" class="btn btn-default btn-sm">Submit</button>
</form>
