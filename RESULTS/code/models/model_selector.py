def model_selector(model, device):

    # Get original dimensions
    #width, height = image_size
    #patch_size = 9

    # Compute padding needed (pad on right and bottom only)
    #pad_width = (patch_size - width % patch_size) % patch_size
    #pad_height = (patch_size - height % patch_size) % patch_size

    # Pad in (left, top, right, bottom) format
    #padding = (int(pad_width/2), int(pad_height/2), int(pad_width/2), int(pad_height/2))

    if model == "resnet":
        from models.resnet_architecture import ResNet50BEVModel
        print("Using RESNET as Encoder")
        rnae = ResNet50BEVModel().to(device)
        base_model = rnae.encoder
        projection_dimension = 512
        return base_model, projection_dimension

    elif model == "dino":
        from models.dino import Dino
        print("Using DINO as Encoder and Max Pooling as Local Feature Aggregation")
        base_model = Dino( device=device)
        projection_dimension = 1536
        return base_model, projection_dimension

    elif model == "dino3":
        from models.dino3 import Dino3
        print("Using DINOv3 as Encoder and Mean-Standard Pooling as Local Feature Aggregation")
        base_model = Dino3(device=device)
        projection_dimension = 1536
        return base_model, projection_dimension


    elif model == "dinovlad":    
        from models.dino2vlad import DinoVlad
        print("Using DINO+NetVLAD as Encoder")
        dinovlad_model = DinoVlad( device=device)
        projection_dimension = 24576 #768*32
        #projection_dimension = 1024
        return dinovlad_model, projection_dimension
    

    elif model == "dino3vlad":    
        from models.dino3vlad import DinoVlad
        print("Using DINO3+NetVLAD as Encoder")
        dinovlad_model = DinoVlad( device=device)
        projection_dimension = 24576 #768*32
        #projection_dimension = 1024
        return dinovlad_model, projection_dimension
    
    elif model == "pc_dinovlad":


        from models.ppdinovlad import PPDinoVlad
        print("Using DINO+NetVLAD as Encoder")
        ppdinovlad_model = PPDinoVlad( device=device)
        projection_dimension = 24576 #768*32
        #projection_dimension = 1024
    
    

        return ppdinovlad_model, projection_dimension

    else:
        raise ValueError(f"Model '{model}' is not supported.")  
    
    
