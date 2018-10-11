{% extends "base.tpl" %}
{% block content %}
    <form action="/experiment" method="post" enctype="multipart/form-data">
        <div class="form-group row mb-0">
            <label for="path_train" class="col-sm-5 col-form-label">Path to training file</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="A CSV containing your training data. Make sure to prepare it first using the prepare tab, above.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="file" class="form-control form-control-sm" id="path_train" name="path_train" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="path_test" class="col-sm-5 col-form-label">Path to test file</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="A CSV containing your test data. Make sure to prepare it first using the prepare tab, above.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="file" class="form-control form-control-sm" id="path_test" name="path_test" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="path_param" class="col-sm-5 col-form-label">Path to parameter file</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="A CSV containing the parameters of your model. The format is 'origin,destination,positive,negative', with one connection per line.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="file" class="form-control form-control-sm" id="path_param" name="path_param">
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="rla" class="col-sm-5 col-form-label">RLA</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The resting level activation (RLA) of all layers that do not use RLA scaling. Layers that do use RLA scaling are scaled between 0 and RLA.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="number" class="form-control form-control-sm" id="rla" value="-0.05" step="0.01" min="-1.0" max=".0" name="rla" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="step" class="col-sm-5 col-form-label">Step size</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The step size of the model. All updates of the model are scaled by this amount. Decreasing the step size increases the granularity of the model, but also increases the run-time by a factor of 1/step_size.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="number" class="form-control form-control-sm" id="step" value="1.0" step="0.1" min=".1" max="1.0" name="step" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="decay" class="col-sm-5 col-form-label">Decay</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The decay rate used in the update equations of the model. A higher decay rate means that nodes are more prone to returning to their resting state.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="number" class="form-control form-control-sm" id="decay" value="0.07" step="0.01" min="0.01" max="1.0" name="decay" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="min" class="col-sm-5 col-form-label">Minimum activation</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The minimum activation to which nodes can drop. This should be lower than the RLA.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="number" class="form-control form-control-sm" id="min" value="-0.02" min="-1.0" step=".01" max="1.0" name="min" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="max" class="col-sm-5 col-form-label">Maximum cycles</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The maximum cycles for which a model can run. Any model which has not reached the threshold will receive a cycle time of -1, allowing you to differentiate between successful and unsuccessful recognitions.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="number" class="form-control form-control-sm" id="max" value="350" min="1" max="100000" name="max" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="threshold" class="col-sm-5 col-form-label">Threshold</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The threshold for recognition. If any node in all monitor layers reaches this threshold, the model has recognized an item, and will return the current cycle time as the recognition time.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="number" class="form-control form-control-sm" id="threshold" value=".7" name="threshold" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="monitorlayers" class="col-sm-5 col-form-label">Monitor Layers</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The names of the layers which are monitored using the threshold above. All layer names should be separated by a space.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="text" class="form-control form-control-sm" id="monitorlayers" value="orthography" name="monitorlayers" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="rlalayers" class="col-sm-5 col-form-label">RLA Layers</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The names of the layers for which to perform RLA scaling. All layer names should be separated by a space.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="text" class="form-control form-control-sm" id="rlalayers" value="orthography" name="rlalayers" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="rlavars" class="col-sm-5 col-form-label">RLA Variable</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The variable by which to scale the RLA for all layers which use RLA scaling, defined above.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="text" class="form-control form-control-sm" id="rlavars" value="frequency" name="rlavars" value="frequency" required>
            </div>
        </div>
        <div class="form-group row mb-0">
            <label for="w" class="col-sm-5 col-form-label">Weight adaptation</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="If this is checked, the model performs weight adaptation by dividing the weights by the number of slots in the input.">
                ?
            </button>
            <div class="col-sm-5">
                <input type="checkbox" class="form-control form-control-sm" id="w" value="w" checked name="w">
            </div>
        </div>
        {% if validation %}
        <div class="form-group row mb-0">
            <label for="feature_set" class="col-sm-5 col-form-label">Errors</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top"  data-html="true" data-original-title="These are the error messages thrown by the app on your input.">
                ?
            </button>
            <div class="col-sm-5">
                <label class="col-xs-5 col-form-label error">{{ validation }}</label>
            </div>
        </div>
        {% endif %}
        <button type="submit" class="btn btn-default btn-sm">Submit</button>
    </form>
{% endblock %}
