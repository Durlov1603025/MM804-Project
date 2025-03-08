import os
import glob
import vtk
import dash
import dash_vtk
import nibabel as nib
import numpy as np
from dash import html, dcc, Input, Output, State
from dash_vtk.utils import to_mesh_state, to_volume_state
import dash_uploader as du
import vtk.util.numpy_support as vtk_np

# Configuration
UPLOAD_FOLDER = "E:/MM 804/Project/tmp"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = dash.Dash(__name__)
du.configure_upload(app, UPLOAD_FOLDER)

global_state = {
    'volume_actor': None,
    'iso_actor': None,
    'valid_state': False
}

app.layout = html.Div([
    html.Div(
        du.Upload(id='uploader', text='Upload CT/MRI NIfTI File'),
        style={'padding': '20px', 'border': '2px dashed #999'}
    ),
    
    html.Div(id='controls', hidden=True, children=[
        html.Div([
            html.H3("3D Transformation Controls", style={'marginBottom': '15px'}),
            
            html.Div([
                html.Div([
                    html.Label("Scale Factor:", style={'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='scale',
                        options=[{'label': f'{x}x', 'value': x} for x in [0.5, 1.0, 1.5, 2.0]],
                        value=1.0,
                        clearable=False,
                        style={'width': '150px'}
                    ),
                ], style={'marginRight': '30px'}),
                
                html.Div([
                    html.Label("Rotation X:", style={'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='rotate-x',
                        options=[{'label': f'{x}°', 'value': x} for x in [0, 45, 90, 180]],
                        value=0,
                        clearable=False,
                        style={'width': '150px'}
                    ),
                ], style={'marginRight': '30px'}),
                
                html.Div([
                    html.Label("Translation X:", style={'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='translate-x',
                        options=[{'label': str(x), 'value': x} for x in [0, 50, 100, 150]],
                        value=0,
                        clearable=False,
                        style={'width': '150px'}
                    ),
                ]),
            ], style={'display': 'flex', 'marginBottom': '25px'}),
            
            dash_vtk.View(
                id='viewport',
                children=[
                    dash_vtk.VolumeRepresentation(
                        id='vol-rep',
                        children=[dash_vtk.VolumeController()]
                    ),
                    dash_vtk.GeometryRepresentation(
                        id='surf-rep',
                        property={'color': [1, 0.5, 0.5], 'opacity': 0.8}
                    )
                ],
                style={
                    'width': '100%',
                    'height': '70vh',
                    'border': '1px solid #ddd'
                }
            )
        ], style={'padding': '25px'})
    ])
])

def create_volume_actor(image_data):
    try:
        print("Initializing volume actor...")
        
        # 1. Validate input data
        if image_data.GetNumberOfPoints() == 0:
            raise ValueError("Empty VTK image data")
        image_data.ComputeBounds()  # Force bounds calculation

        # 2. Create transfer functions
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(-1024, 0.0, 0.0, 0.0)  # Air
        color_func.AddRGBPoint(-500, 0.8, 0.4, 0.4)   # Lung
        color_func.AddRGBPoint(40, 0.9, 0.8, 0.7)     # Soft tissue
        color_func.AddRGBPoint(400, 1.0, 1.0, 0.9)    # Bone

        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(-1024, 0.0)
        opacity_func.AddPoint(-500, 0.1)
        opacity_func.AddPoint(40, 0.3)
        opacity_func.AddPoint(400, 0.9)

        # 3. Configure volume properties
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()

        # 4. Create and configure mapper with GPU support
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(image_data)
        volume_mapper.SetRequestedRenderModeToGPU()
        volume_mapper.Update()  # Force pipeline execution

        # 5. Create volume actor with validation
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(volume_mapper)
        volume_actor.SetProperty(volume_property)
        
        # 6. Final validation before return
        if not volume_actor.GetMapper().GetInput():
            raise ValueError("Mapper has no input data")
        volume_actor.GetMapper().GetInput().ComputeBounds()
        
        print(f"Actor bounds: {volume_actor.GetBounds()}")
        print("Volume actor fully initialized")
        return volume_actor
        
    except Exception as e:
        print(f"Volume actor creation failed: {str(e)}")
        return None

def create_surface_actor(image_data, threshold=400):
    try:
        # Create surface extraction pipeline
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(image_data)
        marching_cubes.SetValue(0, threshold)
        marching_cubes.Update()

        # Create surface mapper
        surf_mapper = vtk.vtkPolyDataMapper()
        surf_mapper.SetInputConnection(marching_cubes.GetOutputPort())
        surf_mapper.ScalarVisibilityOff()

        # Create surface actor with validation
        surf_actor = vtk.vtkActor()
        surf_actor.SetMapper(surf_mapper)
        surf_actor.GetProperty().SetColor(1, 0.5, 0.5)
        surf_actor.GetProperty().SetOpacity(0.8)
        
        if not surf_actor.GetMapper() or not surf_actor.GetProperty():
            raise ValueError("Surface actor components missing")
            
        return surf_actor
    except Exception as e:
        print(f"Surface actor error: {str(e)}")
        return None

def validate_vtk_data(vtk_data):
    """Ensure VTK data meets rendering requirements"""
    if not vtk_data:
        raise ValueError("Null VTK data")
    if vtk_data.GetNumberOfPoints() == 0:
        raise ValueError("Empty VTK dataset")
    if not vtk_data.GetPointData().GetScalars():
        raise ValueError("Missing scalar data")
    return True

# @app.callback(
#     Output('controls', 'hidden'),
#     Output('vol-rep', 'children'),
#     Output('surf-rep', 'children'),
#     Input('uploader', 'isCompleted'),
#     State('uploader', 'fileNames')
# )

# def handle_upload(uploaded, filenames):
#     global_state['valid_state'] = False
    
#     if not uploaded or not filenames:
#         return True, [], []

#     try:
#         # File handling and validation
#         search_path = os.path.join(UPLOAD_FOLDER, "**", filenames[0])
#         matching_files = glob.glob(search_path, recursive=True)
#         if not matching_files:
#             raise FileNotFoundError(f"File not found: {filenames[0]}")
            
#         file_path = matching_files[0]
#         print(f"Loading: {file_path}")

#         # Data loading and validation
#         img = nib.load(file_path)
#         data = img.get_fdata() if hasattr(img, 'get_fdata') else img.get_data()
#         if data is None or data.size == 0:
#             raise ValueError("Empty data array")

#         # VTK data conversion
#         vtk_data = vtk.vtkImageData()
#         vtk_data.SetDimensions(data.shape)
#         vtk_data.SetSpacing(*img.header.get_zooms()[:3])

#         # Convert numpy array with proper memory layout
#         numpy_array = np.ascontiguousarray(data.flatten(order='F'), dtype=np.float32)
#         vtk_array = vtk_np.numpy_to_vtk(numpy_array, deep=True, array_type=vtk.VTK_FLOAT)
#         vtk_array.SetName('intensity')
#         vtk_data.GetPointData().SetScalars(vtk_array)

#         # Actor creation with validation
#         vol_actor = create_volume_actor(vtk_data)
#         surf_actor = create_surface_actor(vtk_data)
        
#         if not vol_actor or not surf_actor:
#             raise ValueError("Actor creation failed")

#         # State conversion with null checks
#         vol_state = to_volume_state(vol_actor)
#         surf_mesh = surf_actor.GetMapper().GetInput()
#         surf_state = to_mesh_state(surf_mesh)

#         if not vol_state or not surf_state:
#             raise ValueError("State conversion failed")

#         # Update global state
#         global_state.update({
#             'volume_actor': vol_actor,
#             'iso_actor': surf_actor,
#             'valid_state': True
#         })

#         return False, [dash_vtk.Volume(state=vol_state)], [dash_vtk.Mesh(state=surf_state)]

#     except Exception as e:
#         print(f"Upload failed: {str(e)}")
#         return True, [], []

@app.callback(
    Output('controls', 'hidden'),
    Output('vol-rep', 'children'),
    Output('surf-rep', 'children'),
    Input('uploader', 'isCompleted'),
    State('uploader', 'fileNames')
)
def handle_upload(uploaded, filenames):
    global_state['valid_state'] = False
    
    if not uploaded or not filenames:
        return True, [], []

    try:
        # =====================================================================
        # 1. File Path Validation
        # =====================================================================
        search_path = os.path.join(UPLOAD_FOLDER, "**", filenames[0])
        matching_files = glob.glob(search_path, recursive=True)
        if not matching_files:
            raise FileNotFoundError(f"File not found: {filenames[0]}")
            
        file_path = matching_files[0]
        print(f"\n=== Loading File ===")
        print(f"Path: {file_path}")
        print(f"File size: {os.path.getsize(file_path)/1024/1024:.2f} MB")

        # =====================================================================
        # 2. NIfTI Data Loading
        # =====================================================================
        img = nib.load(file_path)
        img = nib.as_closest_canonical(img)  # Standardize orientation
        data = img.get_fdata()
        
        print(f"\n=== Data Validation ===")
        print(f"Original shape: {img.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Value range: {np.min(data):.1f} to {np.max(data):.1f}")

        if len(data.shape) != 3:
            raise ValueError(f"3D data required. Got {len(data.shape)}D array")
        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
        if np.any(np.isinf(data)):
            raise ValueError("Data contains infinite values")

        # =====================================================================
        # 3. VTK Data Conversion
        # =====================================================================
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(data.shape)
        vtk_data.SetSpacing(*img.header.get_zooms()[:3])

        # Convert with Fortran ordering and validate
        numpy_array = np.ascontiguousarray(data.flatten(order='F'), dtype=np.float32)
        if numpy_array.nbytes == 0:
            raise ValueError("Flattened array is empty")
            
        vtk_array = vtk_np.numpy_to_vtk(numpy_array, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName('intensity')
        vtk_data.GetPointData().SetScalars(vtk_array)
        vtk_data.Modified()  # Force pipeline update

        print("\n=== VTK Validation ===")
        print(f"VTK dimensions: {vtk_data.GetDimensions()}")
        print(f"VTK spacing: {vtk_data.GetSpacing()}")
        print(f"Scalar range: {vtk_data.GetScalarRange()}")

        # =====================================================================
        # 4. Actor Creation & Validation
        # =====================================================================
        vol_actor = create_volume_actor(vtk_data)
        if not vol_actor or not vol_actor.GetMapper():
            raise ValueError("Volume actor creation failed - check mapper/properties")

        surf_actor = create_surface_actor(vtk_data)
        if not surf_actor or not surf_actor.GetMapper():
            raise ValueError("Surface actor creation failed - check mesh extraction")

        # =====================================================================
        # 5. State Conversion with Enhanced Validation
        # =====================================================================
        print("\n=== State Conversion ===")
        
        # Force VTK pipeline execution
        vol_actor.GetMapper().Update()
        vol_actor.GetMapper().GetInput().ComputeBounds()

        # Convert volume actor state with validation
        try:
            vol_state = to_volume_state(vol_actor)
        except Exception as e:
            print(f"Volume conversion error: {str(e)}")
            raise ValueError("to_volume_state failed") from e
        
        # Validate volume state structure
        if vol_state is None:
            raise ValueError("Volume state is None - check VTK actor initialization")
        if not isinstance(vol_state, dict):
            raise ValueError(f"Volume state should be dict, got {type(vol_state)}")
        
        required_volume_keys = {
            'vtkClass': str,
            'mapper': dict,
            'property': dict,
            'bounds': list
        }

        for key, expected_type in required_volume_keys.items():
            if key not in vol_state:
                raise ValueError(f"Missing key in vol_state: {key}")
            if not isinstance(vol_state[key], expected_type):
                raise ValueError(f"Invalid type for {key}: {type(vol_state[key])} (expected {expected_type})")
            
        print("Volume state validation passed:")
        print(f"- vtkClass: {vol_state['vtkClass']}")
        print(f"- Bounds: {vol_state['bounds']}")


        # Convert surface actor

        try:
            surf_mesh = surf_actor.GetMapper().GetInput()
            surf_state = to_mesh_state(surf_mesh)
        except Exception as e:
            print(f"Surface conversion error: {str(e)}")
            raise ValueError("to_mesh_state failed") from e

        if not surf_state or 'points' not in surf_state or 'cells' not in surf_state:
            print(f"Invalid surface state structure: {surf_state.keys() if surf_state else 'None'}")
            raise ValueError("Surface state missing required components")

        print("Surface state validation passed:")
        print(f"- Points: {len(surf_state['points'])} vertices")
        print(f"- Cells: {len(surf_state['cells'])} polygons")
        
        
        # # Validate surface state
        # if not surf_state or 'points' not in surf_state or 'cells' not in surf_state:
        #     print(f"Invalid surface state: {surf_state}")
        #     raise ValueError("Surface state conversion failed")

        # =====================================================================
        # 6. Update Global State
        # =====================================================================
        global_state.update({
            'volume_actor': vol_actor,
            'iso_actor': surf_actor,
            'valid_state': True
        })

        return False, [dash_vtk.Volume(state=vol_state)], [dash_vtk.Mesh(state=surf_state)]

    except Exception as e:
        print(f"\n=== UPLOAD FAILED ===")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("Traceback:", exc_info=True)
        return True, [], []

@app.callback(
    Output('vol-rep', 'children', allow_duplicate=True),
    Output('surf-rep', 'children', allow_duplicate=True),
    Input('scale', 'value'),
    Input('rotate-x', 'value'),
    Input('translate-x', 'value'),
    prevent_initial_call=True
)
def update_transformations(scale, rot_x, trans_x):
    if not global_state['valid_state']:
        return [], []

    try:
        # Create transformation matrix
        transform = vtk.vtkTransform()
        transform.Scale(scale, scale, scale)
        transform.RotateX(rot_x)
        transform.Translate(trans_x, 0, 0)

        # Apply transformations
        global_state['volume_actor'].SetUserTransform(transform)
        global_state['iso_actor'].SetUserTransform(transform)

        # Get updated states
        vol_state = to_volume_state(global_state['volume_actor'])
        surf_mesh = global_state['iso_actor'].GetMapper().GetInput()
        surf_state = to_mesh_state(surf_mesh)

        return [dash_vtk.Volume(state=vol_state)], [dash_vtk.Mesh(state=surf_state)]
    
    except Exception as e:
        print(f"Transform error: {str(e)}")
        return [], []

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)