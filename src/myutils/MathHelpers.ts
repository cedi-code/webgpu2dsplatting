
export function sigmoid(x : number, offset? : number) : number {
    return 1.0 / (1.0 + Math.exp(-x + (offset ?? 0.0)));
}