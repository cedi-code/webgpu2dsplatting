import { 
    type UniformMember,
    type UniformBufferDescriptor,
    type UniformType,
    type VertexBufferDescriptor,
    type VertexAttribute,
    type BufferDescriptor 
} from "../mytypes/BufferDescriptors";

interface BufferManager {

    device?: GPUDevice;

    init(device: GPUDevice): void;

    createBuffer(descriptor : BufferDescriptor): GPUBuffer;
}

function getByteSize(format: GPUVertexFormat | UniformType): number {
    switch(format) {
        case "uint16":
            return 2;
        case "uint32":
        case "sint32":
        case "float32":
        case "f32":
        case "i32":
        case "u32":
            return 4;
        case "uint32x2":
        case "sint32x2":
        case "float32x2":
        case "vec2f":
        case "vec2i":
        case "vec2u":
            return 8;
        case "uint32x3":
        case "sint32x3":
        case "float32x3":
        case "vec3f":
        case "vec3i":
        case "vec3u":
            return 12;
        case "uint32x4":
        case "sint32x4":
        case "float32x4":
        case "vec4f":
        case "vec4i":
        case "vec4u":
        case "mat2x2f":
            return 16;
        case "mat3x3f":
            return 48;
        case "mat4x4f":
            return 64;
        default:
            throw new Error(`Unsupported GPUVertexFormat: ${format}`);
    }
}

// ARRAY IS NOT SUPPORTED YET
//  UniformType = "f32" | "i32" | "u32" | "vec2f" | "vec3f" | "vec4f" | "mat4x4f" | "mat3x3f" | "mat2x2f" | "vec2i" | "vec3i" | "vec4i" | "vec2u" | "vec3u" | "vec4u";
function alignFromType(type: UniformType): number {
    switch(type) {
        case "f32":
        case "i32":
        case "u32":
            return 4;
        case "vec2f":
        case "vec2i":
        case "vec2u":
        case "mat2x2f":
            return 8;
        case "vec3f":
        case "vec4f":
        case "vec3i":
        case "vec4i":
        case "vec3u":
        case "vec4u":
        case "mat3x3f":
        case "mat4x4f":
            return 16;
        default:
            throw new Error(`Unsupported UniformType: ${type}`);
    }
}


class VertexBufferDescriptorBuilder {

    result : Partial<VertexBufferDescriptor> = {};

    constructor(label: string, count : number, stepMode?: "vertex" | "instance") {
        this.result.label = label;
        this.result.count = count;
        this.result.stepMode = stepMode || "vertex";

    }

    add(shaderLocation: number, name: string, format: GPUVertexFormat): VertexBufferDescriptorBuilder {
        if(!this.result.attributes) {
            this.result.attributes = [];
        }
        // vertex buffer does not need padding
        const local_offset = getByteSize(format);
        const attrib_count = this.result.attributes.length;
        let offset = 0;
        if(attrib_count > 0) {
            const last_attrib = this.result.attributes[attrib_count -1];
            offset = last_attrib.offsetBytes + getByteSize(last_attrib.format);
        }
        
        this.result.attributes.push({
            shaderLocation: shaderLocation,
            name: name,
            format: format,
            offsetBytes: offset,
            offset: offset / 4,
        });
        return this;
    }

    build(): VertexBufferDescriptor {
        const attrib_count = this.result.attributes?.length;
        if(!this.result.label || !this.result.attributes?.length || !attrib_count || attrib_count === 0 || !this.result.stepMode || !this.result.count) {
            throw new Error("Incomplete VertexBufferDescriptor");
        }
        // calc stride
        let stride = this.result.attributes[attrib_count -1].offsetBytes + 
            getByteSize(this.result.attributes[attrib_count -1].format);

        if (stride % 4 !== 0) {
            console.warn(`Vertex Buffer Stride ${stride} is not a multiple of 4?, padding to align.`);
            stride = Math.ceil(stride / 4) * 4;
        }

        return {
            label: this.result.label,
            sizeBytes: stride * this.result.count,
            usage: GPUBufferUsage.VERTEX,
            size: this.result.count * (stride / 4),
            count: this.result.count,
            unitSize: stride / 4,
            unitSizeBytes: stride,
            stepMode: this.result.stepMode,
            attributes: this.result.attributes,
        };
    }
    buildLayout(): GPUVertexBufferLayout {
        const descriptor = this.build();
        return {
            arrayStride: descriptor.unitSizeBytes,
            stepMode: descriptor.stepMode,
            attributes: descriptor.attributes.map((attr : VertexAttribute) => ({
                shaderLocation: attr.shaderLocation,
                format: attr.format,
                offset: attr.offsetBytes,
            })),
        };
    }
}


// ARRAY IS NOT SUPPORTED YET
class UniformBufferDescriptorBuilder {

    result : Partial<UniformBufferDescriptor> = {};

    constructor(label: string, usage?: "uniform" | "storage") {
        this.result.label = label;
        this.result.usage = usage == "storage" ? GPUBufferUsage.STORAGE : GPUBufferUsage.UNIFORM;
    }

    add(name: string, type: UniformType): UniformBufferDescriptorBuilder {
        if(!this.result.attributes) {
            this.result.attributes = [];
        }
        const offset = 0; // offset will be calculated in build()
        this.result.attributes.push({
            name,
            type: type,
            offsetBytes: offset,
            offset: offset / 4,
        });
        return this;
    }

    build(): UniformBufferDescriptor {
        if(!this.result.label || !this.result.attributes || !this.result.usage) {
            throw new Error("Incomplete UniformBufferDescriptor");
        }
        // calc size
        let currOffset = 0;
        let maxAlign = 4;

        // 1. increase offset such that it fits the align
        // 2. assign offset to member
        // 3. increase offset by  byteSize (and not algin)

        // calculating offsetBytes
        this.result.attributes.forEach((member : UniformMember) => {
            const align = alignFromType(member.type);
            currOffset = Math.ceil(currOffset / align) * align;
            member.offsetBytes = currOffset;
            member.offset = currOffset / 4;
            currOffset += getByteSize(member.type);
            
            maxAlign = Math.max(maxAlign, align);

        });

        // padding in the end
        currOffset = Math.ceil(currOffset / maxAlign) * maxAlign;

        return {
            label: this.result.label,
            attributes: this.result.attributes,
            usage: this.result.usage,
            sizeBytes: currOffset,
            size: currOffset / 4,
        };
    }
}


class IndexBufferDescriptorBuilder {

    result : Partial<BufferDescriptor> = {};

    constructor(label: string, count: number, format: GPUIndexFormat) {
        this.result.label = label;
        this.result.usage = GPUBufferUsage.INDEX;
        this.result.sizeBytes = getByteSize(format) * count;
    }

    build(): BufferDescriptor {
        if(!this.result.label || !this.result.usage || !this.result.sizeBytes) {
            throw new Error("Incomplete IndexBufferDescriptor");
        }
        return {
            label: this.result.label,
            sizeBytes: this.result.sizeBytes,
            usage: this.result.usage,
        };
    }
}

const bufferManager: BufferManager = {

    init(device: GPUDevice): void {
        this.device = device;
    },

    createBuffer(descriptor : BufferDescriptor): GPUBuffer {
        if(!this.device) {
            throw new Error("BufferManager not initialized with device");
        }
        const buffer = this.device.createBuffer({
            label: descriptor.label,
            size: descriptor.sizeBytes,
            usage: descriptor.usage | GPUBufferUsage.COPY_DST,
        });
        return buffer;
    }
};

export { bufferManager, VertexBufferDescriptorBuilder, UniformBufferDescriptorBuilder, IndexBufferDescriptorBuilder };