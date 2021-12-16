from datasets.ipn import IPN

def get_training_set(opt, spatial_transform, temporal_transform,target_transform):
    if opt.train_validate:
        subset = ['training', 'validation']
    else:
        subset = 'training'
    
    training_data = IPN(
        opt.video_path,
        opt.annotation_path,
        subset,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration,
        modality=opt.modality)
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,target_transform):
    validation_data = IPN(
        opt.video_path,
        opt.annotation_path,
        'validation',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration,
        modality=opt.modality)
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    if opt.test_subset == 'val':
        subset = 'validation'
    else:
        subset = 'testing'
    test_data = IPN(
        opt.video_path,
        opt.annotation_path, 
        'validation',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        sample_duration=opt.sample_duration,
        modality=opt.modality)

    return test_data
