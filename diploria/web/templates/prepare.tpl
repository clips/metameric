{% extends "base.tpl" %}
{% block content %}
    <form action="/prepare" method="post" enctype="multipart/form-data">
        <div class="form-group row mb-0">
            <label for="path_train" class="col-sm-5 col-form-label">Path to training file</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="A CSV containing the data you want to prepare. The names in Decomposable layers, below, need to be present in the CSV.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="file" class="form-control form-control-sm" id="path_train" name="path_train" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="decomp_layer" class="col-sm-5 col-form-label">Decomposable Layers</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The layers to decompose into separate symbols. Decomposable layers are assumed to consist of symbols which are put into slots, and which are separate.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="text" value="orthography" class="form-control form-control-sm" id="decomp_layer" name="decomp_layer" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="decomp_name" class="col-sm-5 col-form-label">Names of decomposable Layers</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The names given to the decomposed layers above.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="text" value="letters" class="form-control form-control-sm" id="decomp_layer" name="decomp_name" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="feature_layer" class="col-sm-5 col-form-label">Feature layers</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The names of the layers which to decompose into features. Separate by spaces. These layers need to be decomposed layers.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="text" value="letters" class="form-control form-control-sm" id="feature_layer" name="feature_layer" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="feature_set" class="col-sm-5 col-form-label">Feature sets</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top"  data-html="true" data-original-title="The feature sets to use for the corresponding feature layers defined above. Separate by spaces. There need to be as many feature sets as feature names. The currently supported feature sets are:<br>letters: 'fourteen', 'sixteen'<br>phonemes: 'plunkett', 'patpho_bin'.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="text" value="fourteen" class="form-control form-control-sm" id="feature_set" name="feature_set" required>
            </div>
        </div>
        <button type="submit" class="btn btn-default btn-sm">Submit</button>
    </form>
{% endblock %}
