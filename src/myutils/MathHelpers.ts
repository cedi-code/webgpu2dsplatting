import { vec2, type Vec2 } from "wgpu-matrix";
export function randr(low: number, high: number) : number {
    return Math.random()*(high-low) + low; 
}

export function randVec2(low: number, high : number) : Vec2 {
    return vec2.create(randr(low, high), randr(low, high));
}

export function sigmoid(x : number, offset? : number) : number {
    return 1.0 / (1.0 + Math.exp(-x + (offset ?? 0.0)));
}