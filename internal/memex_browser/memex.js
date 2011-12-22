

// The function to flip the bb
function flip_bb(bb,imdim) {
  W = bb[2] - bb[0] + 1;
  H = bb[3] - bb[1] + 1;
  
  bb[2] = imdim[1]-bb[0];
  bb[0] = bb[2]-W+1;
  return bb;
}


// Function to add the raphael image visualization into the element named divid
function show_image(divid,divid2,curid,ext,objectid,bb,imdim,color) {
var href = curid + '.' + objectid + '.html';

im = curid+ext;
elem = document.getElementById(divid);

var paper = Raphael(divid2, imdim[1], imdim[0]);
srcim = datadir + '/' + im;
var c = paper.image(srcim, 0, 0, imdim[1], imdim[0]);

// if we want to clip it
//c.attr({"clip-rect": "1,1,50,50"});

// flip the image which has flip turned on
if (bb[6] == 1) {
  flipstring = " <bold>FLIP</bold>";
  c.scale(-1,1);//.attr({opacity: .5});
  bb = flip_bb(bb,imdim);
} else {
  flipstring = "";
}

c.node.onclick = function() {
  location.href=href;
}

document.getElementById(divid).innerHTML = '<center><h3>' + divid + ' curid='+ curid + '.' + objectid + ' ' + '</h3>Score = ' + bb[11] + flipstring +'</center>';

var w = Math.round(bb[2] - bb[0] + 1);
var h = Math.round(bb[3] - bb[1] + 1);

var c = paper.rect(Math.round(bb[0]), Math.round(bb[1]), w, h);

c.attr({fill: color, stroke: color, "fill-opacity": 0, "stroke-width": 10, cursor: "move"});

// Make the drawn rectangle clickable
c.node.onclick = function() {
location.href=href;
}

}

function show_image_maxos(divid,divid2,curid,ext,objectid,bb,imdim,color,maxos,maxclass) {
var href = curid + '.' + objectid + '.html';

im = curid+ext;
elem = document.getElementById(divid);

var paper = Raphael(divid2, imdim[1], imdim[0]);
srcim = datadir + '/' + im;
var c = paper.image(srcim, 0, 0, imdim[1], imdim[0]);

// flip the image which has flip turned on
if (bb[6] == 1) {
  flipstring = " <bold>FLIP</bold>";
  c.scale(-1,1);//.attr({opacity: .5});
  bb = flip_bb(bb,imdim);
} else {
  flipstring = "";
}

c.node.onclick = function() {
  location.href=href;
}

document.getElementById(divid).innerHTML = '<center><h3>' + divid + ' curid='+ curid + '.' + objectid + ' ' + '</h3>Score = ' + bb[11] + flipstring +' Maxos='+maxos+'</center>';

var w = Math.round(bb[2] - bb[0] + 1);
var h = Math.round(bb[3] - bb[1] + 1);

var c = paper.rect(Math.round(bb[0]), Math.round(bb[1]), w, h);

c.attr({fill: color, stroke: color, "fill-opacity": 0, "stroke-width": 10, cursor: "move"});

// Make the drawn rectangle clickable
c.node.onclick = function() {
location.href=href;
}

}


// function show_image_href(divid,divid2,curid,ext,bb,imdim,color,href) {
// im = curid + ext;
// srcim = datadir + '/' + im;

// var paper = Raphael(divid2, imdim[1], imdim[0]);

// var c = paper.image(srcim, 0, 0, imdim[1], imdim[0]);
// c.node.onclick = function() {
// location.href=href;
// }


// if (bb[6] == 1) {
//   flipstring = " FLIP";
//   c.scale(-1,1);//.attr({opacity: .5});
//   bb = flip_bb(bb,imdim);
// } else {
//   flipstring = "";
// }

// document.getElementById(divid).innerHTML = '<center>Score = ' + bb[11] + flipstring + '</center>';

// c.attr({cursor: "hand"});


// var w = Math.round(bb[2] - bb[0] + 1);
// var h = Math.round(bb[3] - bb[1] + 1);

// var c = paper.rect(Math.round(bb[0]), Math.round(bb[1]), w, h);

// c.attr({fill: color, stroke: color, "fill-opacity": 0, "stroke-width": 10});
// c.node.onclick = function() {
// location.href=href;
// }
// c.attr({cursor: "hand"});
// }


// function show_image2(im,bb) {
// srcim = datadir + '/' + im;

// document.write('Score = ' + bb[11]);
// document.write('<br/>');
// document.write('flip = ' + bb[6]);
// document.write('<br/>');
// document.write('<img src="'+srcim+'" alt="'+im+'" title="Test Image" />');

// }
