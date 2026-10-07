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
    const offset = this.pos;
    const value = this.varint();
    if (!Number.isSafeInteger(value) || value < 8 || value > 0xffffffff) {
      throw new Error(`Invalid protobuf tag at byte ${offset}`);
    }
    const tag = { field: Math.floor(value / 8), wire: value & 7, offset };
    this.lastTag = tag;
    return tag;
  }

  varint() {
    let value = 0;
    for (let index = 0; index < 10; index += 1) {
      if (this.eof()) throw new Error("Unexpected end of protobuf varint");
      const byte = this.bytes[this.pos++];
      if (index === 9 && byte > 1) throw new Error("Invalid protobuf varint");
      value += (byte & 0x7f) * 2 ** (index * 7);
      if ((byte & 0x80) === 0) return value;
    }
    throw new Error("Invalid protobuf varint");
  }

  signedVarint() {
    let value = 0n;
    for (let index = 0; index < 10; index += 1) {
      if (this.eof()) throw new Error("Unexpected end of protobuf varint");
      const byte = this.bytes[this.pos++];
      if (index === 9 && byte > 1) throw new Error("Invalid signed protobuf integer");
      value |= BigInt(byte & 0x7f) << BigInt(index * 7);
      if ((byte & 0x80) === 0) return Number(BigInt.asIntN(64, value));
    }
    throw new Error("Invalid signed protobuf integer");
  }

  fixed32() {
    this.requireBytes(4);
    const value = this.view.getUint32(this.pos, true);
    this.pos += 4;
    return value;
  }

  float32() {
    this.requireBytes(4);
    const value = this.view.getFloat32(this.pos, true);
    this.pos += 4;
    return value;
  }

  requireBytes(length) {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.end - this.pos) {
      throw new Error(`Truncated protobuf field at byte ${this.pos}`);
    }
  }

  bytesField() {
    const length = this.varint();
    const start = this.pos;
    this.requireBytes(length);
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
      this.requireBytes(8);
      this.pos += 8;
      return;
    }

    if (wire === 2) {
      this.bytesField();
      return;
    }

    if (wire === 5) {
      this.requireBytes(4);
      this.pos += 4;
      return;
    }

    if (wire === 3) {
      const field = this.lastTag.field;
      while (!this.eof()) {
        const nested = this.tag();
        if (nested.wire === 4) {
          if (nested.field !== field) throw new Error("Invalid protobuf group end tag");
          return;
        }
        this.skip(nested.wire);
      }
      throw new Error("Truncated protobuf group");
    }

    if (wire === 4) {
      throw new Error("Unexpected protobuf group end tag");
    }

    throw new Error(`Unsupported protobuf wire type ${wire} at byte ${this.lastTag?.offset ?? this.pos} after field ${this.lastTag?.field ?? "?"}`);
  }
}

export function parseOnnxModel(buffer) {
  const reader = new ProtoReader(buffer);
  let hasGraph = false;
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
    else if (field === 7 && wire === 2) { model.graph = reader.message(parseGraph); hasGraph = true; }
    else if (field === 8 && wire === 2) model.opsets.push(reader.message(parseOpset));
    else reader.skip(wire);
  }

  if (!hasGraph) throw new Error("The file does not contain an ONNX graph");
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
  const graph = { name: "", nodes: [], inputs: [], outputs: [], initializers: [], valueInfo: [] };
  let initializerIndex = 0;

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) graph.nodes.push(reader.message(parseNode));
    else if (field === 2 && wire === 2) graph.name = reader.string();
    else if (field === 5 && wire === 2) {
      initializerIndex += 1;
      try {
        graph.initializers.push(reader.message(parseTensor));
      } catch (error) {
        error.message = `Initializer ${initializerIndex}: ${error.message}`;
        throw error;
      }
    } else if (field === 11 && wire === 2) graph.inputs.push(reader.message(parseValueInfo));
    else if (field === 12 && wire === 2) graph.outputs.push(reader.message(parseValueInfo));
    else if (field === 13 && wire === 2) graph.valueInfo.push(reader.message(parseValueInfo));
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
    else if (field === 20 && wire === 0) attribute.type = reader.varint();
    else if (field === 2 && wire === 5) attribute.value = Number(reader.float32().toPrecision(6));
    else if (field === 3 && wire === 0) attribute.value = reader.signedVarint();
    else if (field === 4 && wire === 2) attribute.value = textDecoder.decode(reader.bytesField());
    else if (field === 5 && wire === 2) attribute.value = reader.message(parseTensor);
    else if (field === 6 && wire === 2) attribute.value = reader.message(parseGraph);
    else if (field === 7 && wire === 2) attribute.value = (attribute.value ?? []).concat(readPackedFloat32(reader.bytesField()));
    else if (field === 7 && wire === 5) attribute.value = [...(attribute.value ?? []), reader.float32()];
    else if (field === 8 && wire === 2) {
      const nested = new ProtoReader(reader.bytesField());
      attribute.value ??= [];
      while (!nested.eof()) attribute.value.push(nested.signedVarint());
    }
    else if (field === 8 && wire === 0) attribute.value = [...(attribute.value ?? []), reader.signedVarint()];
    else if (field === 9 && wire === 2) attribute.value = [...(attribute.value ?? []), reader.string()];
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
    else if ([4, 5, 6, 7, 9, 10, 11].includes(field) && wire === 2) {
      const bytes = reader.bytesField();
      if (field === 4) requireCompleteElements(bytes, 4);
      else if (field === 10) requireCompleteElements(bytes, 8);
      else if (field === 5 || field === 7 || field === 11) {
        const packed = new ProtoReader(bytes);
        while (!packed.eof()) packed.varint();
      }
      // Count serialized data payload, excluding field tags and length prefixes.
      tensor.byteSize += bytes.byteLength;
    } else if ((field === 4 && wire === 5) || (field === 10 && wire === 1) || ((field === 5 || field === 7 || field === 11) && wire === 0)) {
      const start = reader.pos;
      reader.skip(wire);
      tensor.byteSize += reader.pos - start;
    } else reader.skip(wire);
  }

  return tensor;
}

function parseValueInfo(reader) {
  const value = { name: "", type: "tensor", dataType: "unknown", shape: null };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) value.name = reader.string();
    else if (field === 2 && wire === 2) Object.assign(value, reader.message(parseType));
    else reader.skip(wire);
  }

  return value;
}

function parseType(reader) {
  const type = { type: "tensor", dataType: "unknown", shape: null };

  while (!reader.eof()) {
    const { field, wire } = reader.tag();
    if (field === 1 && wire === 2) Object.assign(type, reader.message(parseTensorType));
    else reader.skip(wire);
  }

  return type;
}

function parseTensorType(reader) {
  const tensorType = { type: "tensor", dataType: "unknown", shape: null };

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
  requireCompleteElements(bytes, 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = [];
  for (let offset = 0; offset + 4 <= bytes.byteLength; offset += 4) {
    values.push(Number(view.getFloat32(offset, true).toPrecision(6)));
  }
  return values;
}

function requireCompleteElements(bytes, width) {
  if (bytes.byteLength % width !== 0) throw new Error("Truncated packed protobuf field");
}
