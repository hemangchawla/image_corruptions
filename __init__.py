import cv2
import os
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone import ViewField as F
from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names
import skimage
import numpy as np
from numba import njit

# Adapted from 
# The imagecorruptions library uses older version of skimage where gaussian is applied with multichannel arg.
# It has been replaced with channel_axis

# TODO: Move functions to respective util files
def wrapper_gaussian_filter(image, sigma, multichannel=False, channel_axis=-1):
    if skimage.__version__ >= '0.19':
        # Use channel_axis instead of multichannel
        return skimage.filters.gaussian(image, sigma, channel_axis=channel_axis)
    else:
        # Use multichannel parameter
        return skimage.filters.gaussian(image, sigma, multichannel=multichannel)

@njit()
def _shuffle_pixels_njit_glass_blur(d0,d1,x,c):

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(d0 - c[1], c[1], -1):
            for w in range(d1 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = wrapper_gaussian_filter(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255

def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][
        severity - 1]

    x = np.uint8(
        wrapper_gaussian_filter(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    x = _shuffle_pixels_njit_glass_blur(np.array(x).shape[0],np.array(x).shape[1],x,c)

    return np.clip(wrapper_gaussian_filter(x / 255., sigma=c[0], multichannel=True), 0,
                   1) * 255

def _convert_to_title_case(text):
    return ' '.join([word.capitalize() for word in text.split('_')]).strip()

def _get_target_view(ctx, target):
    
    if target == "selected":
        return ctx.view.select(ctx.selected)

    if target == "entire":
        return ctx.dataset.match_tags("corrupted", bool=False)

    return ctx.view.match_tags("corrupted", bool=False)

def _get_selected_corruptions(ctx):
    selected_corruptions = []
    corruption_type = ctx.params.get("corruption_type", "Common")
    if corruption_type == "Common":
        corruptions = get_corruption_names()
        select_all = ctx.params.get("select_all_common",False) 
    else:
        corruptions = get_corruption_names('validation')
        select_all = ctx.params.get("select_all_validation",False) 

    if not select_all:
        for c in corruptions:
            if ctx.params.get(c, False):
                selected_corruptions.append(c)
    else:
        selected_corruptions = corruptions
    return selected_corruptions

def _get_severities(ctx):
    select_all = ctx.params.get("select_all_severities", False)
    if not select_all:
        return [ctx.params.get("severity", 1)]
    else:
        return range(1,6)

class ImageCorruptions(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="image_corruptions",
            label="Image Corruptions",
            description="Generate corrupted images of your dataset to measure robustness.",
            dynamic=True,
            execute_as_generator=True,
        )
    
    def resolve_input(self, ctx):
        corruptions = get_corruption_names()
        corruptions_validation = get_corruption_names('validation')
        
        inputs = types.Object()

        # Choose Corruption Type
        corruption_types = ["Common", "Validation"]
        corruption_types_group = types.RadioGroup()
        
        for choice in corruption_types:
            corruption_types_group.add_choice(choice, label=choice)
        
        inputs.enum(
            "corruption_type",
            corruption_types_group.values(),
            label="Corruption ",
            description="Choose validation corruptions when augementations similar to common corruptions are used during training.",
            view=types.TabsView(),
            required=False,
        )

        corruption_type = ctx.params.get("corruption_type", "Common")

        if corruption_type == "Common":
        # Choose Common Corruptions
        
            inputs.bool(
                "select_all_common",
                default=False,
                label="Select all",
                view=types.SwitchView(),
            )
    
            select_all_common = ctx.params.get("select_all_common", False)
            
            if not select_all_common:
                for c in corruptions:
                    title = _convert_to_title_case(c)
                    inputs.bool(
                    c,
                    label=title,
                    view=types.CheckboxView(space=3),
                )
        else:
        # Choose Validation
            inputs.bool(
                "select_all_validation",
                default=False,
                label="Select all",
                view=types.SwitchView(),
            )
    
            select_all_validation = ctx.params.get("select_all_validation", False)
    
            if not select_all_validation:
                for c in corruptions_validation:
                    title = _convert_to_title_case(c)
                    inputs.bool(
                    c,
                    label=title,
                    view=types.CheckboxView(space=3),
                )

        # Corruption Severity

        inputs.bool(
            "select_all_severities",
            default=False,
            label="All Severities",
            description="All severities from 1 to 5",
            view=types.SwitchView(),
        )
    
        select_all_severities = ctx.params.get("select_all_severities", False)

        if not select_all_severities:
            inputs.float(
                "severity",
                label="Severity",
                description="Severity of Corruption between 1 and 5",
                view=types.SliderView(componentsProps={'slider': {'min': 1, 'max': 5, 'step': 1}}),
                default=1)

        # Run on dataset/selected samples/ view

        has_view = ctx.view != ctx.dataset.view()
        has_selected = bool(ctx.selected)
        default_target = "entire"
        if has_view or has_selected:
            target_choices = types.RadioGroup()
            target_choices.add_choice(
                "entire",
                label="Entire Original dataset",
                description="Run model on the entire original dataset",
            )
    
            if has_view:
                target_choices.add_choice(
                    "current",
                    label="Current view",
                    description="Run model on the current view",
                )
                default_target = "current"
    
            if has_selected:
                target_choices.add_choice(
                    "selected",
                    label="Selected samples",
                    description="Run model on the selected samples",
                )
                default_target = "selected"
    
            inputs.enum(
                "target",
                target_choices.values(),
                default=default_target,
                view=target_choices,
            )
        else:
            ctx.params["target"] = default_target

        
        
        # Execution Mode
        _execution_mode(ctx, inputs)
        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)
    
    def execute(self, ctx):
        selected_corruptions = _get_selected_corruptions(ctx)

        if len(selected_corruptions) > 0:
            severity = ctx.params.get("severity", 1)
            severities = _get_severities(ctx)
        
        target = ctx.params.get("target", None) 
        target_view = _get_target_view(ctx, target)    

        for i, sample in enumerate(target_view):
            for corruption in selected_corruptions:
                for severity in severities:
                    new_sample = corrupt_sample(sample,corruption, severity)
                
        ctx.trigger("reload_dataset")

def get_new_filepath(sample, corruption, severity):
    image_dir = os.path.dirname(sample.filepath)
    image_dir_base = os.path.basename(os.path.normpath(image_dir))
    parent_dir = os.path.dirname(image_dir)
    sample_dir_path = "/".join(parent_dir.split("/"))
    corrupted_sample_dir_path = os.path.join(os.path.dirname(image_dir), image_dir_base + "_corrupted", f"{corruption}", f"{severity}")
    filename = os.path.basename(sample.filepath)
    new_filepath = os.path.join(corrupted_sample_dir_path, filename)
    return new_filepath

def corrupt_sample(sample, corruption, severity):
    new_filepath = get_new_filepath(sample, corruption, severity)
    os.makedirs(os.path.dirname(new_filepath), exist_ok=True)

    image = cv2.imread(sample.filepath)

    # TODO Create PR to imagecorruptions library to fix from source or create latest version of library
    if corruption == "gaussian_blur":
        corrupted = gaussian_blur(image, severity=severity)
    elif corruption == "glass_blur":
         corrupted = glass_blur(image, severity=severity)
    else:
        corrupted = corrupt(image, corruption_name=corruption, severity=severity)
        
    cv2.imwrite(new_filepath, corrupted)
    
    new_sample = fo.Sample(filepath=new_filepath, 
                           tags=["corrupted"],  corruption_name=corruption, corruption_severity=severity)
    new_sample["original_sample_id"] = sample.id
    sample._dataset.add_sample(new_sample)

def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


def register(plugin):
    plugin.register(ImageCorruptions)
