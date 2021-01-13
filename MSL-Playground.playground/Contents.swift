import MetalKit
import PlaygroundSupport

print("Test")

let frame = NSRect(x: 0, y: 0, width: 640, height: 576)
let delegate = MetalView()
let view = MTKView(frame: frame, device: delegate.device)
view.delegate = delegate
PlaygroundPage.current.liveView = view
