function darkTheme() {
  var heading = document.getElementById("heading");
  var tagline = document.getElementById("tagline");
  var ask = document.getElementById("ask");
  var dark = document.getElementById("dark");
  var light = document.getElementById("light");
  var developer = document.getElementById("developer");
  var dname = document.getElementById("d-name");
  var body = document.getElementById("body");
  var footer = document.getElementById("footer");
  var extrab = document.getElementsByClassName("btn-secondary");
  
  light.style = ('display: block');
  dark.style = ('display: none');
  heading.style = ('color: antiquewhite');
  tagline.style = ('color: antiquewhite');
  developer.style = ('color: antiquewhite');
  dname.style = ('color: antiquewhite')
  body.style = ('background-color: rgb(16, 25, 56)');
  ask.style = ('background-color: rgb(13, 0, 255);');
  footer.style = ('background-color: rgb(16, 25, 56);');
  extrab.style = ('background-color: #5f5f5f;');
}

function lightTheme() {
  var heading = document.getElementById("heading");
  var tagline = document.getElementById("tagline");
  var ask = document.getElementById("ask");
  var dark = document.getElementById("dark");
  var light = document.getElementById("light");
  var dark = document.getElementById("dark");
  var light = document.getElementById("light");
  var developer = document.getElementById("developer");
  var dname = document.getElementById("d-name");
  var body = document.getElementById("body");
  var footer = document.getElementById("footer");
  var extrab = document.getElementsByClassName("btn-secondary");
  
  light.style = ('display: none');
  dark.style = ('display: block');
  heading.style = ('color: black');
  tagline.style = ('color: black');
  developer.style = ('color: black');
  dname.style = ('color: black')
  body.style = ('background-color: rgb(230, 230, 230)');
  ask.style = ('background-color: black');
  footer.style = ('background-color: rgb(230, 230, 230);');
  extrab.style = ('background-color: #86a7ff;');
}

