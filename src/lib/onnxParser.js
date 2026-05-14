const textDecoder = new TextDecoder("utf-8");

const TENSOR_TYPES = {
  1: "float32",
  2: "uint8",
  3: "int8",
  4: "uint16",
  5: "int16",
  6: "int32",
  7: "int64",
  8: "string",
  9: "bool",
  10: "float16",
  11: "float64",
  12: "uint32",
  13: "uint64",
  16: "bfloat16"
};

class ProtoReader {
  constructor(buffer, start = 0, end = buffer.byteLength) {
    this.bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    this.view = new DataView(this.bytes.buffer, this.bytes.byteOffset, this.bytes.byteLength);
    this.pos = start;
    this.end = end;
  }

  eof() {
    return this.pos >= this.end;
  }

  tag() {
    const value = this.varint();
    return { field: value >> 3, wire: value & 7 };
  }

  varint() {
    let value = 0;
    let shift = 0;

    while (this.pos < this.end) {
      const byte = this.bytes[this.pos++];
      value += (byte & 0x7f) * 2 ** shift;
      if ((byte & 0x80) === 0) return value;
      shift += 7;
    }

    throw new Error("Unexpected end of protobuf varint");
  }

  fixed32() {
    const value = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return value;
  }

  float32() {
    const value = this.view.getFloat32(this.pos, true);
    this.pos += 4;
    return value;
  }

  bytesField() {
    const length = this.varint();
    const start = this.pos;
    this.pos += length;
    return this.bytes.subarray(start, start + length);
  }

  string() {
    return textDecoder.decode(this.bytesField());
  }

  message(parser) {
    const bytes = this.bytesField();
    return parser(new ProtoReader(bytes));
  }

  packedVarints() {
    const bytes = this.bytesField();
    const reader = new ProtoReader(bytes);
    const values = [];
    while (!reader.eof()) values.push(reader.varint());
    return values;
  }

  skip(wire) {
    if (wire === 0) {
      this.varint();
      return;
    }

    if (wire === 1) {
      this.pos += 8;
      return;
    }

    if (wire === 2) {
      this.pos += this.varint();
      return;
    }

    if (wire === 5) {
      this.pos += 4;
      return;
    }

    throw new Error(`Unsupported protobuf wire type ${wire}`);
  }
}

export function parseOnnxModel(buffer) {
  const reader = new ProtoReader(buffer);
  const model = {
    irVersion: null,
    producerName: "",
    producerVersion: "",
    opsets: [],
    graph: { name: "", nodes: [], inputs: [], outputs: [], initializers: [] }
  };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();

    if (field === 1 && wire === 0) model.irVersion = reader.varint();
    else if (field === 2 && wire === 2) model.producerName = reader.string();
    else if (field === 3 && wire === 2) model.producerVersion = reader.string();
    else if (field === 7 && wire === 2) model.graph = reader.message(parseGraph);
    else if (field === 8 && wire === 2) model.opsets.push(reader.message(parseOpset));
    else reader.skip(wire);
  }

  return model;
}

function parseOpset(reader) {
  const opset = { domain: "", version: null };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) opset.domain = reader.string();
    else if (field === 2 && wire === 0) opset.version = reader.varint();
    else reader.skip(wire);
  }

  return opset;
}

function parseGraph(reader) {
  const graph = { name: "", nodes: [], inputs: [], outputs: [], initializers: [] };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) graph.nodes.push(reader.message(parseNode));
    else if (field === 2 && wire === 2) graph.name = reader.string();
    else if (field === 5 && wire === 2) graph.initializers.push(reader.message(parseTensor));
    else if (field === 11 && wire === 2) graph.inputs.push(reader.message(parseValueInfo));
    else if (field === 12 && wire === 2) graph.outputs.push(reader.message(parseValueInfo));
    else reader.skip(wire);
  }

  return graph;
}

function parseNode(reader) {
  const node = { name: "", opType: "", domain: "", inputs: [], outputs: [], attributes: [] };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) node.inputs.push(reader.string());
    else if (field === 2 && wire === 2) node.outputs.push(reader.string());
    else if (field === 3 && wire === 2) node.name = reader.string();
    else if (field === 4 && wire === 2) node.opType = reader.string();
    else if (field === 5 && wire === 2) node.attributes.push(reader.message(parseAttribute));
    else if (field === 7 && wire === 2) node.domain = reader.string();
    else reader.skip(wire);
  }

  return node;
}

function parseAttribute(reader) {
  const attribute = { name: "", type: null, value: null };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) attribute.name = reader.string();
    else if (field === 4 && wire === 0) attribute.type = reader.varint();
    else if (field === 5 && wire === 5) attribute.value = Number(reader.float32().toPrecision(6));
    else if (field === 6 && wire === 0) attribute.value = reader.varint();
    else if (field === 7 && wire === 2) attribute.value = textDecoder.decode(reader.bytesField());
    else if (field === 10 && wire === 2) attribute.value = readPackedFloat32(reader.bytesField());
    else if (field === 11 && wire === 2) attribute.value = reader.packedVarints();
    else reader.skip(wire);
  }

  return attribute;
}

function parseTensor(reader) {
  const tensor = { name: "", dataType: "unknown", dims: [], byteSize: 0 };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();

    if (field === 1 && wire === 0) tensor.dims.push(reader.varint());
    else if (field === 1 && wire === 2) tensor.dims.push(...reader.packedVarints());
    else if (field === 2 && wire === 0) tensor.dataType = TENSOR_TYPES[reader.varint()] ?? "unknown";
    else if (field === 8 && wire === 2) tensor.name = reader.string();
    else if (field === 9 && wire === 2) tensor.byteSize += reader.bytesField().byteLength;
    else if ((field === 4 || field === 6 || field === 7) && wire === 2) tensor.byteSize += reader.bytesField().byteLength;
    else {
      if (wire === 5) tensor.byteSize += 4;
      reader.skip(wire);
    }
  }

  return tensor;
}

function parseValueInfo(reader) {
  const value = { name: "", type: "tensor", dataType: "unknown", shape: [] };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) value.name = reader.string();
    else if (field === 2 && wire === 2) Object.assign(value, reader.message(parseType));
    else reader.skip(wire);
  }

  return value;
}

function parseType(reader) {
  const type = { type: "tensor", dataType: "unknown", shape: [] };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) Object.assign(type, reader.message(parseTensorType));
    else reader.skip(wire);
  }

  return type;
}

function parseTensorType(reader) {
  const tensorType = { type: "tensor", dataType: "unknown", shape: [] };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 0) tensorType.dataType = TENSOR_TYPES[reader.varint()] ?? "unknown";
    else if (field === 2 && wire === 2) tensorType.shape = reader.message(parseShape);
    else reader.skip(wire);
  }

  return tensorType;
}

function parseShape(reader) {
  const shape = [];

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) shape.push(reader.message(parseDimension));
    else reader.skip(wire);
  }

  return shape;
}

function parseDimension(reader) {
  let value = "?";

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 0) value = reader.varint();
    else if (field === 2 && wire === 2) value = reader.string();
    else reader.skip(wire);
  }

  return value;
}

function readPackedFloat32(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = [];
  for (let offset = 0; offset + 4 <= bytes.byteLength; offset += 4) {
    values.push(Number(view.getFloat32(offset, true).toPrecision(6)));
  }
  return values;
}
