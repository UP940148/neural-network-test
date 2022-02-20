/** @module matrix */

/**
 * Represents a Matrix.
 * @class
 * @property {Array} data - Stored values.
 */
export class Matrix {
  /**
   * Create new Matrix.
   * @constructor
   * @param {Integer} rows - Number of rows.
   * @param {Integer} columns - Number of columns.
   * @param {Number} val - Value of each entry.
   */
  constructor(rows, columns, val) {
    this.data = [];
    const row = [];
    for (let c = 0; c < columns; c++) {
      row.push(val);
    }
    for (let r = 0; r < rows; r++) {
      this.data.push([...row]);
    }
    if (val === undefined) {
      this.randomise();
    }
  }

  /**
   * Display matrix row by row in the console.
   * @returns {Undefined}
   */
  show() {
    for (let i = 0; i < this.data.length; i++) {
      console.log(this.data[i]);
    }
  }

  /**
   * Get dimensions of matrix.
   * @returns {Array} [rows, columns]
   */
  size() {
    return [this.data.length, this.data[0].length];
  }

  /**
   * Set the value in a given position.
   * @param {Integer} row - Row number.
   * @param {Integer} column - Column number.
   * @param {Number} value - Value to set.
   */
  set(row, column, value) {
    this.data[row][column] = value;
  }

  /**
   * Get the value in a given position.
   * @param {Integer} row - Row number.
   * @param {Integer} column - Column number.
   * @returns {Number}
   */
  get(row, column) {
    return this.data[row][column];
  }

  /**
   * Multiply this matrix with another.
   * @param {Matrix} matB - Matrix to multiply with.
   * @returns {Matrix}
   */
  mult(matB) {
    // If A's column count != B's row count, return false
    if (this.data[0].length !== matB.data.length) {
      return false;
    }
    // Set i and j equal to A's rows, and B's columns respectively
    const r = this.data.length;
    const c = matB.data[0].length;

    // Set n to be the number of iterations per value
    const n = this.data[0].length;

    // Create new matrix
    const result = new Matrix(r, c);

    /*
      Apply multiplication

      A's rows * B's columns

      For every entry in A's row, multiply by respective entry in B's column
    */

    // For every row of new matrix
    for (let i = 0; i < r; i++) {
      // For every column of new matrix
      for (let j = 0; j < c; j++) {
        let total = 0;
        for (let k = 0; k < n; k++) {
          // Add the product of the two relevant entries, to the current total
          total += this.get(i, k) * matB.get(k, j);
        }
        // Set the total in the correct position of the resulting matrix
        result.set(i, j, total);
      }
    }

    return result;
  }

  /**
   * Add a matrix to this matrix.
   * @param {Matrix} matB - Matrix to add.
   * @returns {Matrix}
   */
  add(matB) {
    // If matrices aren't same size, return false
    const aSize = this.size();
    const bSize = matB.size();
    if (aSize[0] !== bSize[0] || aSize[1] !== bSize[1]) {
      return false;
    }
    const [rows, columns] = this.size();
    const result = new Matrix(rows, columns);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        const value = this.get(i, j) + matB.get(i, j);
        result.set(i, j, value);
      }
    }

    return result;
  }

  subtract(matB) {
    // If matrices aren't same size, return false
    const aSize = this.size();
    const bSize = matB.size();
    if (aSize[0] !== bSize[0] || aSize[1] !== bSize[1]) {
      return false;
    }
    const [rows, columns] = this.size();
    const result = new Matrix(rows, columns);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) {
        const value = this.get(i, j) - matB.get(i, j);
        result.set(i, j, value);
      }
    }

    return result;
  }

  scalarDiv(divisor) {
    const [rows, columns] = this.size();
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        const result = this.get(r, c) / divisor;
        this.set(r, c, result);
      }
    }
  }

  scalarMult(multiplier) {
    const [rows, columns] = this.size();
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        const result = this.get(r, c) * multiplier;
        this.set(r, c, result);
      }
    }
  }

  transpose() {
    const [rows, columns] = this.size();
    const other = new Matrix(columns, rows, 0);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        other.set(c, r, this.get(r, c));
      }
    }
    this.data = other.data;
  }

  randomise() {
    const [rows, columns] = this.size();
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        this.set(r, c, (Math.random() * 10 - 5));
      }
    }
  }

  apply(f) {
    const [rows, columns] = this.size();
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        const result = f(this.get(r, c));
        this.set(r, c, result);
      }
    }
  }

  /**
   * Returns the Hadamard Product of two matrices.
   * @param {Matrix} matB - Multiplier matrix.
   * @returns {Matrix} Result of product.
   */
  hadamard(matB) {
    // Get matrix dimensions
    const [rows, columns] = this.size();
    const [bRows, bColumns] = matB.size();

    // If dimensions don't match, return false
    if (rows !== bRows || columns !== bColumns) {
      return false;
    }

    const result = new Matrix(rows, columns);
    // For every value, get product
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        const product = this.get(r, c) * matB.get(r, c);
        result.set(r, c, product);
      }
    }

    // Return resulting matrix
    return result;
  }

  clone(matB) {
    this.data = [...matB.data];
    for (let i = 0; i < this.data.length; i++) {
      this.data[i] = [...matB.data[i]];
    }
  }
}
