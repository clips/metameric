%rebase('templates/base.tpl', title='Index')
<img src="static/content/plot.png" alt="No graph yet." class="center"/>
<form action="/analysis_2" method="post" enctype="multipart/form-data">
    <div class="grid">
        % for layer in inputs:
            <label for="{{layer}}">{{layer}}</label>
            <input id=itrain type=button value="?" class="btn btn-default btn-sm"/>
            <input type="text" id="{{layer}}" name="{{layer}}"/>
        % end
    </div>
    <button type="submit" class="btn btn-default btn-sm">Submit</button>
</form>
