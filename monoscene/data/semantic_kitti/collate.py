import torch


def collate_fn(batch):
    data = {}
    imgs = []
    CP_mega_matrices = []
    targets = []
    frame_ids = []
    sequences = []

    cam_ks = []
    T_velo_2_cams = []
    frustums_masks = []
    frustums_class_dists = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())

        if "frustums_masks" in input_dict:
            frustums_masks.append(torch.from_numpy(input_dict["frustums_masks"]))
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        img = input_dict["img"]
        imgs.append(img)

        frame_ids.append(input_dict["frame_id"])
        sequences.append(input_dict["sequence"])
        
        
        target = torch.from_numpy(input_dict["target"])
        targets.append(target)
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))            

    ret_data = {
        "frame_id": frame_ids,
        "sequence": sequences,
        "frustums_class_dists": frustums_class_dists,
        "frustums_masks": frustums_masks,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "img": torch.stack(imgs),
        "CP_mega_matrices": CP_mega_matrices,
        "target": torch.stack(targets)
    }
    

    for key in data:
        ret_data[key] = data[key]
    return ret_data


def sequential_collate_fn(batch):
    """
    SequentialKittiDataset에 맞게 collate function을 수정.
    batch의 각 요소는 [dict1, dict2, ..., dictN] (sequence_length 개수) 형태임.
    """
    data = {}
    imgs = []
    CP_mega_matrices = []
    targets = []
    frame_ids = []
    sequences = []

    cam_ks = []
    T_velo_2_cams = []
    frustums_masks = []
    frustums_class_dists = []

    scale_3ds = batch[0][0]["scale_3ds"]  # 첫 번째 시퀀스의 첫 번째 데이터에서 scale_3ds 가져옴
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    # batch = [ [dict1, dict2, ..., dictN], [dict1, dict2, ..., dictN], ... ]
    sequence_length = len(batch[0])  # 각 샘플이 가진 시퀀스 길이

    for seq_idx in range(sequence_length):  # 시퀀스 길이만큼 반복
        imgs_seq = []
        CP_mega_matrices_seq = []
        targets_seq = []
        cam_ks_seq = []
        T_velo_2_cams_seq = []
        frustums_masks_seq = []
        frustums_class_dists_seq = []

        frame_ids_seq = []
        sequences_seq = []

        for batch_idx, input_dict in enumerate(batch):  # 배치 내 모든 시퀀스를 반복
            seq_data = input_dict[seq_idx]  # 현재 프레임(seq_idx)의 데이터 가져오기

            cam_ks_seq.append(torch.from_numpy(seq_data["cam_k"]).double())
            T_velo_2_cams_seq.append(torch.from_numpy(seq_data["T_velo_2_cam"]).float())

            if "frustums_masks" in seq_data:
                frustums_masks_seq.append(torch.from_numpy(seq_data["frustums_masks"]))
                frustums_class_dists_seq.append(
                    torch.from_numpy(seq_data["frustums_class_dists"]).float()
                )

            for key in data:
                if seq_idx == 0:  # 첫 번째 시퀀스에서 리스트 초기화
                    data[key] = []
                data[key].append(torch.from_numpy(seq_data[key]))

            imgs_seq.append(seq_data["img"])
            frame_ids_seq.append(seq_data["frame_id"])
            sequences_seq.append(seq_data["sequence"])

            target = torch.from_numpy(seq_data["target"])
            targets_seq.append(target)
            CP_mega_matrices_seq.append(torch.from_numpy(seq_data["CP_mega_matrix"]))

        # 스택 쌓기 (시퀀스 단위)
        imgs.append(torch.stack(imgs_seq))  # (batch_size, seq_length, C, H, W)
        CP_mega_matrices.append(torch.stack(CP_mega_matrices_seq))  # (batch_size, seq_length, 64, 64, 4)
        targets.append(torch.stack(targets_seq))  # (batch_size, seq_length, 256, 256, 32)

        cam_ks.append(torch.stack(cam_ks_seq))  # (batch_size, seq_length, 3, 3)
        T_velo_2_cams.append(torch.stack(T_velo_2_cams_seq))  # (batch_size, seq_length, 4, 4)

        if frustums_masks_seq:
            frustums_masks.append(torch.stack(frustums_masks_seq))  # (batch_size, seq_length, ...)
            frustums_class_dists.append(torch.stack(frustums_class_dists_seq))  # (batch_size, seq_length, ...)

        frame_ids.append(frame_ids_seq)
        sequences.append(sequences_seq)

    # 최종 데이터 딕셔너리 반환
    ret_data = {
        "frame_id": frame_ids,  # (batch_size, seq_length)
        "sequence": sequences,  # (batch_size, seq_length)
        "frustums_class_dists": frustums_class_dists if frustums_masks else None,
        "frustums_masks": frustums_masks if frustums_masks else None,
        "cam_k": cam_ks,  # (batch_size, seq_length, 3, 3)
        "T_velo_2_cam": T_velo_2_cams,  # (batch_size, seq_length, 4, 4)
        "img": torch.stack(imgs),  # (batch_size, seq_length, C, H, W)
        "CP_mega_matrices": torch.stack(CP_mega_matrices),  # (batch_size, seq_length, 64, 64, 4)
        "target": torch.stack(targets),  # (batch_size, seq_length, 256, 256, 32)
    }

    for key in data:
        ret_data[key] = torch.stack(data[key])  # (batch_size, seq_length, ...)

    return ret_data
