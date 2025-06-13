import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO

# Set page config
st.set_page_config(page_title="Enhanced Pixel Art Generator", layout="wide")

# App title
st.title("Pixel Art Generator")
st.markdown("Create **ultra-crisp**, high-quality pixel art with advanced processing!")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image using PIL
    image = Image.open(uploaded_file)
    
    # Convert to RGB if needed
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    
    # Show original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, caption=f"Size: {image.width}×{image.height}", use_container_width=True)
    
    # Get original dimensions
    orig_width, orig_height = image.size
    
    # Enhanced pixelation parameters
    st.subheader("Pixelation Controls")
    
    # Quality enhancement options
    with st.expander("Quality Enhancement Options", expanded=True):
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            enhance_contrast = st.checkbox("Enhance Contrast", value=True, help="Increases contrast for better pixel definition")
            enhance_colors = st.checkbox("Color Enhancement", value=True, help="Boosts color saturation for vibrant pixels")
            sharpen_pre = st.checkbox("Pre-sharpen", value=False, help="Sharpens image before pixelation")
        
        with col_q2:
            color_quantization = st.checkbox("Color Quantization", value=True, help="Reduces colors for authentic pixel art look")
            if color_quantization:
                num_colors = st.slider("Color Palette Size", min_value=8, max_value=256, value=64, step=8)
            edge_preserve = st.checkbox("Edge Preservation", value=True, help="Better preserves important edges")
    
    # Two modes: Simple and Advanced
    mode = st.radio("Choose control mode:", ["Simple Mode", "Advanced Mode"], horizontal=True)
    
    if mode == "Simple Mode":
        st.markdown("**Quick presets for perfect pixel art:**")
        preset = st.selectbox(
            "Choose a preset:",
            ["Custom", "Game Sprite (16×16)", "Icon (32×32)", "Small Art (64×64)", "Medium Art (128×128)", "Large Art (256×256)"]
        )
        
        if preset == "Custom":
            pixel_width = st.slider("Pixel Art Width", min_value=8, max_value=512, value=90, step=2)
            pixel_height = st.slider("Pixel Art Height", min_value=8, max_value=512, value=120, step=2)
        else:
            size_map = {
                "Game Sprite (16×16)": (16, 16),
                "Icon (32×32)": (32, 32),
                "Small Art (64×64)": (64, 64),
                "Medium Art (128×128)": (128, 128),
                "Large Art (256×256)": (256, 256)
            }
            pixel_width, pixel_height = size_map[preset]
            st.info(f"Selected: {pixel_width}×{pixel_height} pixels")
        
        # Calculate optimal pixel size for display
        display_width = min(pixel_width * 6, 1300)
        display_height = min(pixel_height * 6, 1300)
        
    else:  # Advanced Mode
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Pixel Art Dimensions:**")
            pixel_width = st.number_input(
                "Pixel Art Width (in pixels)",
                min_value=8,
                max_value=512,
                value=90,
                step=2,
                help="This is the actual pixel count in your pixel art"
            )
            pixel_height = st.number_input(
                "Pixel Art Height (in pixels)",
                min_value=8,
                max_value=512,
                value=120,
                step=2,
                help="This is the actual pixel count in your pixel art"
            )
        
        with col_b:
            st.markdown("**Display Options:**")
            pixel_size = st.slider(
                "Pixel Size Multiplier", 
                min_value=1, 
                max_value=25, 
                value=6, 
                step=1,
                help="How big each pixel appears in the final image"
            )
            
            display_width = min(pixel_width * pixel_size, 1300)
            display_height = min(pixel_height * pixel_size, 1300)
            
            show_pixel_grid = st.checkbox("Show Pixel Grid", value=False, help="Adds grid lines between pixels")
    
    # Show current settings
    st.info(f"Pixel Art: {pixel_width}×{pixel_height} pixels | Display: {display_width}×{display_height} pixels")
    
    # Enhanced pixelation function
    def enhance_image_quality(img, enhance_contrast, enhance_colors, sharpen_pre):
        """Apply quality enhancements to the image"""
        enhanced_img = img.copy()
        
        if sharpen_pre:
            # Apply unsharp mask for better detail preservation
            enhanced_img = enhanced_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        if enhance_contrast:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced_img)
            enhanced_img = enhancer.enhance(1.3)
        
        if enhance_colors:
            # Enhance color saturation
            enhancer = ImageEnhance.Color(enhanced_img)
            enhanced_img = enhancer.enhance(1.2)
        
        return enhanced_img
    
    def quantize_colors(img, num_colors):
        """Reduce the number of colors in the image for authentic pixel art look"""
        # Convert to P mode with specified number of colors
        quantized = img.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
        # Convert back to RGB
        return quantized.convert('RGB')
    
    def smart_resize_with_edge_preservation(img, target_width, target_height, preserve_edges=True):
        """Advanced resizing with edge preservation"""
        if preserve_edges and min(target_width, target_height) >= 16:
            # For larger pixel art, use a two-step process
            # Step 1: Resize to 2x target size with high quality
            intermediate_w = target_width * 2
            intermediate_h = target_height * 2
            
            # Use different resampling based on size reduction factor
            original_pixels = img.width * img.height
            target_pixels = intermediate_w * intermediate_h
            reduction_factor = original_pixels / target_pixels
            
            if reduction_factor > 16:
                # Heavy reduction - use LANCZOS
                intermediate = img.resize((intermediate_w, intermediate_h), Image.Resampling.LANCZOS)
            else:
                # Moderate reduction - use BICUBIC for smoother results
                intermediate = img.resize((intermediate_w, intermediate_h), Image.Resampling.BICUBIC)
            
            # Step 2: Resize to final size with box filter for pixel-perfect results
            final = intermediate.resize((target_width, target_height), Image.Resampling.BOX)
        else:
            # For small pixel art, use direct high-quality resize
            final = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return final
    
    def add_pixel_grid(img, pixel_size):
        """Add subtle grid lines between pixels"""
        if pixel_size < 4:  # Only add grid for larger pixels
            return img
            
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Add vertical grid lines
        for x in range(0, width, pixel_size):
            if x < width:
                img_array[:, x:x+1] = img_array[:, x:x+1] * 0.9  # Darken grid lines
        
        # Add horizontal grid lines
        for y in range(0, height, pixel_size):
            if y < height:
                img_array[y:y+1, :] = img_array[y:y+1, :] * 0.9  # Darken grid lines
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def create_enhanced_pixel_art(img, target_width, target_height, display_width, display_height, 
                                enhance_contrast, enhance_colors, sharpen_pre, color_quantization, 
                                num_colors, edge_preserve, show_grid=False, pixel_size=1):
        """Create enhanced pixel art with advanced quality processing"""
        
        # Step 1: Enhance the original image
        enhanced_img = enhance_image_quality(img, enhance_contrast, enhance_colors, sharpen_pre)
        
        # Step 2: Smart resize to pixel art dimensions
        pixel_art = smart_resize_with_edge_preservation(enhanced_img, target_width, target_height, edge_preserve)
        
        # Step 3: Apply color quantization if enabled
        if color_quantization:
            pixel_art = quantize_colors(pixel_art, num_colors)
        
        # Step 4: Scale up to display size using nearest neighbor for crisp pixels
        if display_width != target_width or display_height != target_height:
            display_img = pixel_art.resize((display_width, display_height), Image.Resampling.NEAREST)
        else:
            display_img = pixel_art.copy()
        
        # Step 5: Add pixel grid if requested
        if show_grid and 'show_pixel_grid' in locals() and show_pixel_grid:
            display_img = add_pixel_grid(display_img, pixel_size)
        
        return pixel_art, display_img
    
    # Generate the enhanced pixel art
    show_grid = 'show_pixel_grid' in locals() and show_pixel_grid if mode == "Advanced Mode" else False
    pixel_art, display_img = create_enhanced_pixel_art(
        image, pixel_width, pixel_height, display_width, display_height,
        enhance_contrast, enhance_colors, sharpen_pre, color_quantization,
        num_colors if color_quantization else 256, edge_preserve, show_grid,
        pixel_size if mode == "Advanced Mode" else 6
    )
    
    # Show pixelated image
    with col2:
        st.subheader("Enhanced Pixel Art Result")
        st.image(
            display_img, 
            caption=f"Pixel Art: {pixel_width}×{pixel_height} | Display: {display_width}×{display_height}",
            use_container_width=False
        )
    
    # Quality comparison
    if st.checkbox("Show Quality Comparison", value=False):
        st.subheader("Quality Comparison")
        
        # Create a basic version for comparison
        basic_pixel_art = image.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
        basic_display = basic_pixel_art.resize((display_width, display_height), Image.Resampling.NEAREST)
        
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            st.image(basic_display, caption="Basic Pixelation", use_container_width=True)
        with col_comp2:
            st.image(display_img, caption="Enhanced Pixelation", use_container_width=True)
    
    # Download options
    st.subheader("Download Options")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        # Download original pixel art (small file)
        buf_original = BytesIO()
        pixel_art.save(buf_original, format="PNG", optimize=True)
        byte_original = buf_original.getvalue()
        
        st.download_button(
            label=f"Pixel Art ({pixel_width}×{pixel_height})",
            data=byte_original,
            file_name=f"enhanced_pixel_art_{pixel_width}x{pixel_height}.png",
            mime="image/png",
            help="Original pixel art - small file size"
        )
    
    with col_d2:
        # Download display version (larger file)
        buf_display = BytesIO()
        display_img.save(buf_display, format="PNG", optimize=True)
        byte_display = buf_display.getvalue()
        
        st.download_button(
            label=f"Display Version ({display_width}×{display_height})",
            data=byte_display,
            file_name=f"enhanced_display_{display_width}x{display_height}.png",
            mime="image/png",
            help="Scaled version for viewing/printing"
        )
    
    with col_d3:
        # Download ultra-high quality version
        uhq_width = min(pixel_width * 10, 2000)
        uhq_height = min(pixel_height * 10, 2000)
        uhq_img = pixel_art.resize((uhq_width, uhq_height), Image.Resampling.NEAREST)
        
        buf_uhq = BytesIO()
        uhq_img.save(buf_uhq, format="PNG", optimize=True)
        byte_uhq = buf_uhq.getvalue()
        
        st.download_button(
            label=f"Ultra-HQ ({uhq_width}×{uhq_height})",
            data=byte_uhq,
            file_name=f"ultra_hq_pixel_art_{uhq_width}x{uhq_height}.png",
            mime="image/png",
            help="Ultra high-quality version for professional use"
        )
    
    # Show enhanced stats
    st.subheader("Enhanced Statistics")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        st.metric("Original Size", f"{orig_width}×{orig_height}")
    with col_s2:
        st.metric("Pixel Art Size", f"{pixel_width}×{pixel_height}")
    with col_s3:
        reduction_factor = (orig_width * orig_height) / (pixel_width * pixel_height)
        st.metric("Size Reduction", f"{reduction_factor:.1f}x")
    with col_s4:
        if color_quantization:
            st.metric("Color Palette", f"{num_colors} colors")
        else:
            st.metric("Colors", "Full spectrum")

else:
    st.info("Please upload an image to start creating enhanced pixel art!")
    
    # Show enhanced features
    st.subheader("Enhanced Features:")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("""
        **Quality Enhancements:**
        - Advanced contrast enhancement
        - Color saturation boosting
        - Pre-processing sharpening
        - Smart edge preservation
        - Professional color quantization
        """)
    
    with col_f2:
        st.markdown("""
        **Advanced Processing:**
        - Multi-step resize algorithm
        - Optimal resampling selection
        - Pixel grid overlay option
        - Ultra-high quality exports
        - Quality comparison tools
        """)
    
    st.markdown("""
    **Perfect for:** Game development, NFT art, retro designs, social media, professional pixel art projects
    """)