$("#qtrain").click (function (event)    // Open button Treatment
{
  if ($("#dialog").dialog ("isOpen")) alert ("Already open !");
  else $("#dialog").dialog ("open");
});


$("div#dialog").dialog ({
  autoOpen : false
});
