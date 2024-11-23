########################################################
# args for users
# Attention!!: pred should have the same name as gt

pred_path="/data/data1/test/2024_02_04_20_04_12"
gt_path="/data/data1/spacenet/gt"

# mode_list=("apls topo")
mode_list=("apls")
########################################################
echo pred_path: $pred_path
echo gt_path: $gt_path

for mode in ${mode_list[@]}; do
    echo "Calculating ${mode}"
    # pixel-level
    if [[ $mode == "pixel-level" ]]; then
        python pixel_level_metric.py "${pred_path}/pred_centerline" "${gt_path}/gt_centerline" "${pred_path}/metric_save"
    fi

    # ecm
    if [[ $mode == "ecm" ]]; then
        python ecm.py "${pred_path}/pred_centerline" "${gt_path}/gt_centerline" "${pred_path}/metric_save"
    fi

    # apls
    if [[ $mode == "apls" ]]; then
        if [ ! -d "${pred_path}/metric_save/apls_results" ]; then
            mkdir -p "${pred_path}/metric_save/apls_results"
        fi

        count=0
        mark=''
        for graph_name in $(ls "${pred_path}/pred_graph"); do
            count=$(($count + 1))
            # if [ $(($count % 1)) -eq 0 ]; then
            #     printf "progress:[%-80s]%d\r" "${mark}" "${count}"
            #     mark="##${mark}"
            # fi
            if [ $(($count % 1)) -eq 0 ]; then
                printf "progress:[%-80s]%d\r" "${mark}" "${count}"
                mark="##${mark}"
            fi
            if [ -e "$gt_path/gt_graph/$graph_name" ]; then
                save_txt_name=${graph_name%pickle}txt
                if [ ! -e "${pred_path}/metric_save/apls_results/$save_txt_name" ]; then
                    # only compute apls if didn't compute before
                    python ./apls/convert.py "${pred_path}/pred_graph/$graph_name" "${pred_path}/metric_save/pred.json"
                    python ./apls/convert.py "$gt_path/gt_graph/$graph_name" "${pred_path}/metric_save/gt.json"
                    go run ./apls/apls.go "${pred_path}/metric_save/gt.json" "${pred_path}/metric_save/pred.json" "${pred_path}/metric_save/apls_results/$save_txt_name"
                    rm "${pred_path}/metric_save/pred.json"
                    rm "${pred_path}/metric_save/gt.json"
                fi

            else
                echo "${pred_path}/pred_graph/$graph_name"
            fi
        done
        python ./apls/compute_average_apls.py "${pred_path}/metric_save/apls_results" "${pred_path}/metric_save"
    fi

    # topo
    if [[ $mode == "topo" ]]; then
        if [ ! -d "${pred_path}/metric_save/topo_results" ]; then
            mkdir -p "${pred_path}/metric_save/topo_results"
        fi

        count=0
        mark=''
        for graph_name in $(ls "${pred_path}/pred_graph"); do
            count=$(($count + 1))
            # if [ $(($count % 1)) -eq 0 ]; then
            #     printf "progress:[%-80s]%d\r" "${mark}" "${count}"
            #     mark="##${mark}"
            # fi
            if [ $(($count % 100)) -eq 0 ]; then
                printf "progress:[%-80s]%d\r" "${mark}" "${count}"
                mark="##${mark}"
            fi
            if [ -e "$gt_path/gt_graph/$graph_name" ]; then
                save_txt_name=${graph_name%pickle}txt
                if [ ! -e "${pred_path}/metric_save/topo_results/$save_txt_name" ]; then
                    python ./topo/main.py -graph_gt "${gt_path}/gt_graph/$graph_name" -graph_prop "${pred_path}/pred_graph/$graph_name" -output "${pred_path}/metric_save/topo_results/$save_txt_name"
                fi
            else
                echo "${pred_path}/pred_graph/$graph_name"
            fi
        done
        python ./topo/compute_average_topo.py "${pred_path}/metric_save/topo_results" "${pred_path}/metric_save"
    fi

done
