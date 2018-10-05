<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
    <meta name="description" content="">
    <meta name="author" content="">
    <link rel="icon" href="../../favicon.ico">

    <title>Diploria: an Interactive Activation Simulator.</title>
    <script src="/static/scripts/jquery-3.3.1.min.js"></script>
    <script src="/static/scripts/bootstrap.bundle.min.js"></script>
    <script src="/static/scripts/validation.js"></script>
    <!-- Bootstrap core CSS -->
    <link rel="stylesheet" type="text/css" href="/static/content/bootstrap.min.css" />
    <!-- Custom styles for this template -->
    <link rel="stylesheet" type="text/css" href="/static/content/site.css" />
  </head>
  <!-- Fixed navbar -->
  <div class="container" id="topContent" bg-primary>
      <a href="https://www.uantwerpen.be/en/research-groups/clips/">
          <img src="static/content/clips_logo.png" alt="logo" class="pull-left" id="logo">
      </a>
      <img src="static/content/ChromisNiloticus.jpg" alt="Fish" class="center" name="fish" id="fish">

  </div>

<div class="container">
  <div class="header clearfix">
      <div class="d-flex flex-column flex-md-row align-items-center p-1 mb-1 bg-white">
      <h4 class="my-0 mr-md-auto font-weight-bold">Diploria</h4>
      <nav class="">
        <a href="home" class="mr-md-2">Home</a>
        <a href="experiment" class="mr-md-2">Experiment</a>
        <a href="prepare" class="mr-md-2">Prepare Data</a>
        <a href="analysis" class="mr-md-2">Analysis</a>
        <a href="about" class="mr-md-2">About</a>
      </nav>
    </div>
  </div>

    <div class="container theme-showcase" role="main">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
