export interface BufferDescriptor {
    label: string;
    sizeBytes: number;
    usage: GPUBufferUsageFlags;
}

export interface VertexBufferDescriptor extends BufferDescriptor {
    unitSize: number;
    unitSizeBytes: number;
    size: number;
    count: number;
    stepMode: "vertex" | "instance";
    attributes: VertexAttribute[];
}

export interface UniformBufferDescriptor extends BufferDescriptor {
    attributes: UniformMember[];
    size: number;
}

export type VertexAttribute = {
    shaderLocation: number;
    name: string;
    format: GPUVertexFormat;
    offsetBytes: number;
    offset: number;
};


export type UniformType = "f32" | "i32" | "u32" | "vec2f" | "vec3f" | "vec4f" | "mat4x4f" | "mat3x3f" | "mat2x2f" | "vec2i" | "vec3i" | "vec4i" | "vec2u" | "vec3u" | "vec4u";

export interface UniformMember {
    name: string;
    type: UniformType;
    // The Manager (or a Utility) calculates it based on WGSL rules.
    offsetBytes: number; 
    offset : number;
}