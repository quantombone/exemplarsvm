datadir = "http://balaton.graphics.cs.cmu.edu/sdivvala/.all/Datasets/Pascal_VOC/VOC2007/JPEGImages/";

function show_image(divid,im,bb,imdim,color) {

document.writeln('Score = ' + bb[11]);
document.writeln('<br/>');
var paper = Raphael(divid, imdim[1], imdim[0]);
srcim = datadir + '/' + im;
var c = paper.image(srcim, 0, 0, imdim[1], imdim[0]);

// flip the image which has flip turned on
//if (bb[6] == 1) {
//c.scale(-1,1).attr({opacity: .5});
//}


var w = Math.round(bb[2] - bb[0] + 1);
var h = Math.round(bb[3] - bb[1] + 1);

var c = paper.rect(Math.round(bb[0]), Math.round(bb[1]), w, h);

c.attr({fill: color, stroke: color, "fill-opacity": 0, "stroke-width": 10, cursor: "move"});
}

function show_image2(im,bb) {
srcim = datadir + '/' + im;

document.write('Score = ' + bb[11]);
document.write('<br/>');
document.write('flip = ' + bb[6]);
document.write('<br/>');
document.write('<img src="'+srcim+'" alt="'+im+'" title="Test Image" />');

}
