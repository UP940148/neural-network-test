/** @module network */

import { Matrix } from './matrix.js';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const imgData = [];
const labels = [];



export class NNetwork {
  /**
   * Represents a Neural Network.
   * @constructor
   * @param {Array} sizes - One integer for each layer of the network.
   */
  constructor(sizes) {
    this.weights = [];
    this.activations = [];
    this.Z = [];
    this.biases = [];
    this.layerCount = sizes.length;
    this.structure = sizes;
    this.α = 0.01;
    for (let l = 0; l < sizes.length; l++) { // Using 'l' for 'Layer'
      /*
        Create weight matrices where
        each row corresponds to a neuron in the next layer (l + 1),
        and every column corresponds to a neuron in this layer (l)
      */

      // We don't want to try and create weights after final layer
      if (l < sizes.length - 1) {
        this.weights.push(new Matrix(sizes[l + 1], sizes[l]));
      }

      /*
        Create activation vectors (as single column matrices)
        Initialise activations to a random number
      */

      this.activations.push(new Matrix(sizes[l], 1, 0));
      this.Z.push(new Matrix(sizes[l], 1, 0));

      /*
        Create bias vectors (as single column matrices)
        Initialise biases to 0
      */

      // Layer 0 has no bias
      if (l > 0) {
        this.biases.push(new Matrix(sizes[l], 1));
      }
    }
  }

  /**
   * Calculates the activation of a given layer.
   * @param {Integer} l - Layer number, must be greater than 0.
   */
  calculate(l) {
    if (l < 1) {
      return false;
    }
    // activations[l] = sigmoid(weights[l - 1] * activations[l - 1] + bias[l - 1])
    // a1 = σ(W*a0 + b)
    const wa0 = this.weights[l - 1].mult(this.activations[l - 1]); // Weights multiplied by activations[0]
    const wab = wa0.add(this.biases[l - 1]); // Sum of weights and biases
    this.Z[l].clone(wab);
    wab.apply(sigmoid); // Sigmoid result of weights an biases

    this.activations[l] = wab; // Set activations to result of sigmoided weights and biases
  }

  /**
   * Display Network activations layer by layer in console.
   * @returns {Undefined}
   */
  show() {
    for (let l = 0; l < this.activations.length; l++) {
      console.log(this.activations[l]);
    }
  }

  /**
   * Calculate network activations given a specific input.
   * @param {Array} input - Network input.
   */
  forwardProp(input) {
    const l0Length = this.activations[0].size()[0];

    // Populate Layer 0 activations
    for (let i = 0; i < l0Length; i++) {
      this.activations[0].set(i, 0, imgData[input][i]);
    }
    //this.activations[0].set(5, 0, 255);

    for (let l = 1; l < this.activations.length; l++) {
      this.calculate(l);
    }
  }

  /**
   * Adjust weights and biases of network, given a desired output.
   * @param {Integer} desired - Single digit that the network should have output.
   */
  backProp(index) {
    const desired = labels[index];
    // Get desired value as vector
    // Vector should be same size as output layer, initialised to 0
    const Y = new Matrix(this.structure[this.structure.length - 1], 1, 0);
    // Set expected neuron to 1
    Y.set(desired, 0, 1);

    /*
      Get sensitivity of cost with respect to each weight and bias

      For single Neuron layers:

      wL   = Weight preceeding neuron in layer L

      bL   = Bias applied to neuron in layer L

      zL   = function that sets activation of neuron before sigmoid
           = wL * a[L-1] + bL

      aL   = Sigmoid function
           = σ(zL)

      cO   = Cost function
           = (aL - y)^2

      δcO/δwL = Change in cO with respect to wL
              = δzL/δwL * δaL/δzL * δcO/δaL     -- Chain rule

      δzL/δwL = Change in zL with respect to wL
              = a[L-1]

      δaL/δzL = Change in aL with respect to zL
              = σ'(zL)

      δcO/δaL = Change in cO with respect to aL
              = 2(aL - y)

      δc0/δbL = Change in cO with respect to bL
              = δzL/δbL * δaL/δzL * δcO/δaL     -- Chain rule

      δzL/δbL = Change in zL with respect to bL
              = 1

      δcO/δa[l-1] = Change in cO with respect to a[L-1]
                  = δzL/δa[L-1] * δaL/δzL * δcO/δaL


      Multiple neurons per layer:

      k    = Index of neuron in layer L-1
      j    = Index of neuron in layer L

      wjkL = Weight between neuron k -> j

      bjL  = Bias applied to neuron j in layer L


      zjL = Sum of activations in layer L-1 multiplied by weights connecting to layer L, plus bjL
          = Σ( (wjkL * ak[L-1]) ) + bjL for every k in layer L-1

      ajL = Sigmoid of all zj in layer L
          = σ(zjL)

      cO  = Sum of each individual cost of neurons in layer O
          = Σ( (ajL - yj)^2 ) for every j in layer L

      δcO/δwjkL  = Change in cO with respect to wjkL
                 = δzjL/δwjkL * δajL/δzjL * δcO/δajL -- Chain rule

      δzjL/δwjkL = Change in zjL with respect to wjkL
                 = Σ( ak[L-1] ) for every k in layer L-1

      δajL/δzjL  = Change in ajL with respect to zjL
                 = σ'(zjL)

      δcO/δajL   = Change in cO with respect to ajL
                 = Σ( 2(ajL - yj) ) for every j in layer L

      δcO/δbjL   = Change in cO with respect to bjL
                 = δzjL/δbjL * δajL/δzjL * δcO/δajL -- Chain rule

      δzjL/δbjL  = Change in zjL with respect to bjL
                 = 1


      δcO/δak[L-1] = Change in cO with respect to ak[L-1]
                   = Σ( δzjL/δak[L-1] * δajL/δzjL * δcO/δajL ) for every j in layer L
    */

    // Get network cost
    const cost = this.getCost(Y);
    console.log(cost);
    // Y.show();
    // this.activations[3].show();
    // console.log(cost);
    // console.log('---------------------');

    const layers = this.structure.length;

    // Create lists for matrices
    const CW = []; // Weights L[3]
    const CB = []; // Biases L[3]
    const AZ = []; // dA/dZ
    const X = [undefined];

    // δC/δA3 = 2 * (A[3] - Y)
    const dCdA = this.activations[layers - 1].subtract(Y);
    dCdA.scalarMult(2);

    AZ.push(new Matrix(this.structure[0], 1, 1));
    AZ[0].clone(this.Z[0]);
    AZ[0].apply(sigPrime);

    for (let i = 1; i < layers; i++) {
      CW.push(new Matrix(this.structure[i], this.structure[i - 1], 1));
      CW[i - 1].clone(dCdA);
      CB.push(new Matrix(this.structure[i], 1, 1));
      CB[i - 1].clone(dCdA);
      AZ.push(new Matrix(this.structure[i], 1, 1));
      AZ[i].clone(this.Z[i]);
      AZ[i].apply(sigPrime);
      X.push(undefined);
    }

    for (let l = layers - 1; l >= 0; l--) {
      // console.log('-------------', l);

      if (l === layers - 1) {
        // If top layer, calculate X value and continue
        // XL = 2(AL - Y) ∘ σ'(ZL)  -- Where L is the last layer
        X[l] = this.activations[l].subtract(Y);
        X[l].scalarMult(2);
        X[l] = X[l].hadamard(AZ[l]);
        continue;
      }
      // Else, calculate X value and apply to CW and CB
      // Xl = (WlT * X[l+1]) ∘ σ'(Zl)  -- Where WlT = Wl Transposed
      X[l] = new Matrix(1, 1, 1);
      X[l].clone(this.weights[l]);
      X[l].transpose();
      X[l] = X[l].mult(X[l + 1]);
      X[l] = X[l].hadamard(AZ[l]);

      const aTranspose = new Matrix(1, 1, 1);
      aTranspose.clone(this.activations[l]);
      aTranspose.transpose();
      // console.log('W');
      // console.log(X[l + 1].size());
      // console.log(aTranspose.size());
      CW[l] = X[l + 1].mult(aTranspose);

      const bTranspose = new Matrix(1, 1, 1);
      bTranspose.clone(this.biases[l]);
      // bTranspose.transpose();
      // console.log('B');
      // console.log(X[l + 1].size());
      // console.log(bTranspose.size());
      CB[l] = X[l + 1].hadamard(bTranspose);
    }

    for (let l = 0; l < layers - 1; l++) {
      let scaledW = new Matrix(1, 1, 1);
      scaledW.clone(this.weights[l]);
      scaledW.scalarMult(this.α);
      scaledW = scaledW.hadamard(CW[l]);

      //if (l === 2) {
      //  console.log('- - - - - - - - -');
      //  console.log(this.weights[l].data[5]);
      //}


      this.weights[l] = this.weights[l].subtract(scaledW);

      //if (l === 2) {
      //  console.log(this.weights[l].data[5]);
      //  console.log('- - - - - - - - -');
      //}

      const scaledB = new Matrix(1, 1, 1);
      scaledB.clone(this.biases[l]);
      scaledB.scalarMult(this.α);
      this.biases[l] = this.biases[l].subtract(scaledB.hadamard(CB[l]));
    }
  }

  /**
   * Get the cost of the current network state, given a desired output.
   * @param {Matrix} y - Desired output layer.
   * @return {Number} Network cost.
   */
  getCost(y) {
    let cost = 0;
    const layer = this.structure.length - 1;
    const nCount = this.structure[this.structure.length - 1];
    // For every neuron in output layer
    for (let n = 0; n < nCount; n++) {
      // Apply quadratic cost function to neuron
      let value = this.activations[layer].get(n, 0) - y.get(n, 0);
      value = value * value;
      cost += value;
    }
    // Return average
    cost = cost / nCount;

    return cost;
  }


  train(count) {
    // this.activations[3].show();
    for (let index = 0; index < count; index++) {
      this.forwardProp(0);
      //console.log(this.weights[this.structure.length - 2].data[0])// .show();
      //this.activations[this.structure.length - 1].show();
      //console.log('')
      this.backProp(0);
      //console.log(this.weights[this.structure.length - 2].data[0])// .show();
      //this.activations[this.structure.length - 1].show();
      //console.log('- - - - - - - - - -');
      if (index % 100 === 0) {
        console.log(index, '/', count);
      }
    }
    //console.log('-------------------');
    // this.activations[0].show();
    //console.log(this.weights[this.structure.length - 2].data[0])// .show();
    //this.activations[this.structure.length - 1].show();
  }
}

/**
 * Sigmoid activation function.
 * @param {Number} x - Value to apply Sigmoid function to.
 * @returns {Number}
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigPrime(x) {
  const exponent = Math.exp(-x);
  return exponent / ((exponent + 1) * (exponent + 1));
}


function getTrainingData() {
  // Read data in from files
  const trainingImages = fs.readFileSync(path.join(__dirname, '/training_data/training-images.idx3-ubyte'));
  const trainingLabels = fs.readFileSync(path.join(__dirname, '/training_data/training-labels.idx1-ubyte'));

  // There are 60,000 entries in the training set
  for (let i = 0; i < 60000; i++) {
    // Each pixel is represented as one pair of hex characters
    // Offset is 0016
    // Images are 28px * 28px
    const pixels = [];
    for (let j = 0; j < 28 * 28; j++) {
      const imgOffset = (i * 28 * 28) + 16;
      pixels.push(trainingImages[j + imgOffset]/255);
    }
    imgData.push(pixels);

    const labelOffset = 8;
    labels.push(trainingLabels[i + labelOffset]);
  }
}

getTrainingData();
// console.log(imgData[0]);
// console.log(labels[0]);
// fs.writeFileSync(path.join(__dirname, '/0.txt'), imgData[0].toString());

const net = new NNetwork([784, 512, 256, 128, 10]);
net.train(1000);
net.forwardProp(0);
//net.activations[0].show();
//console.log('')
net.activations[4].show();
