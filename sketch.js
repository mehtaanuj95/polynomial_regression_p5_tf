
let x_vals = [];
let y_vals = [];

let a, b, c, d;
let dragging = false;

var learningRate;
var newRate;

var rangeSlider;
var optimiserValue;
var newOptimiserValue;

var optimizer ;
var can; //Our canvas

function setup() {
  can = createCanvas(400, 400);


  //To prevent mouse clicking outside the canvas
  //https://github.com/processing/p5.js/issues/1400
  can.mousePressed(function() {
    dragging = true;
  });

  can.mouseReleased(function() {
    dragging = false;
  });

  //Initialize variables
  initVariables();

  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));
}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = ax^3 + bx^2 + cx + d
  const ys = xs.pow(tf.scalar(3)).mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d);
  return ys;
}

function draw() {
  
  //Update variables from sliders
  updateVariables();

  if (dragging) {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
  } else {
    tf.tidy(() => {
      if (x_vals.length > 0) {
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => loss(predict(x_vals), ys));
      }
    });
  }

  background(0);

  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }

  const curveX = [];
  for (let x = -1; x <= 1; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();

  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);
    vertex(x, y);
  }
  endShape();

  // console.log(tf.memory().numTensors);
}

function initVariables() {
  rangeSlider = parseInt(document.getElementById('rangeSlider').value);
  learningRate = rangeSlider / 100;

  optimiserValue = $("#optimiserValue").val().toString();
  

  if(optimiserValue == "sgd") {
    optimizer = tf.train.sgd(learningRate);
  }
  else {
    optimizer = tf.train.adam(learningRate);
  }

  newRate = learningRate;
}

function updateVariables() {

  rangeSlider = parseInt(document.getElementById('rangeSlider').value);
  newRate = rangeSlider / 100;
  if(newRate != learningRate) {
    learningRate = newRate;
    
  } 
  newOptimiserValue = $("#optimiserValue").val().toString() ;
  if(newOptimiserValue!= optimiserValue) {
    if(newOptimiserValue == "sgd" ) {
      optimizer = tf.train.sgd(learningRate);
      console.log("sgd");
    }
    else {
      optimizer = tf.train.adam(learningRate);
    }
  }
  optimiserValue = newOptimiserValue;

}

function clearCanvas() {
  console.log("Clearing canvas evenet triggered");
  x_vals = [];
  y_vals = [];
}






/*

tf.train.sgd(learningRate);
tf.train.momentum (learningRate, momentum, useNesterov?)
tf.train.adagrad (learningRate, initialAccumulatorValue?)
tf.train.rmsprop (learningRate, decay?, momentum?, epsilon?, centered?) 
*/