%rebase('templates/base.tpl', title='Index')
<form action="/prepare" method="post" enctype="multipart/form-data">
    <div class="grid">
        <label for="path_train">Input data</label>
        <input id=atrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="file" id="path_train" name="path_train">
        <label for="decomp_layer">Decomposable Layers</label>
        <input id=dtrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="text" id="decomp_layer" name="decomp_layer" value="orthography">
        <label for="decomp_name">Names of decomposed layers</label>
        <input id=dtrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="text" id="decomp_name" name="decomp_name" value="letters">
        <label for="feature_layer">Feature Layers</label>
        <input id=dtrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="text" id="feature_layer" name="feature_layer" value="letters">
        <label for="feature_set">Feature sets</label>
        <input id=dtrain type=button value="?" class="btn btn-default btn-sm"/>
        <input type="text" id="feature_set" name="feature_set" value="fourteen">
    </div>
    <button type="submit" class="btn btn-default btn-sm">Submit</button>
</form>
