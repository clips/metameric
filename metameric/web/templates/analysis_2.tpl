{% extends "base.tpl" %}
{% block content %}
<img src="data:image/png;base64,{{data}}" alt="No graph yet." class="center img-fluid"/>
<form action="/analysis_2" method="post" enctype="multipart/form-data">
    <div class="grid">
        {% for layer in inputs: %}
            <label for="{{layer}}">{{layer}}</label>
            <button type="button" class="btn btn-default btn-sm" data-toggle="tooltip" data-placement="top" data-original-title="The data to input to the model.">
                ?
            </button>
            <input type="text" id="{{layer}}" name="{{layer}}"/>
        {% endfor %}
    </div>
    <button type="submit" class="btn btn-default btn-sm">Submit</button>
</form>
{% endblock %}
