#!/bin/sh

envarg_parse() {
    _bin_name="$1"; shift
    _prefix="$1"; shift
    _defaults_base="$1"; shift
    _suffixes="$1"; shift

    _defaults_path_default="${_defaults_base}.default"
    _shift_c=0
    _do_help="False"
    _do_default_defaults="True"


    while [ $# -gt 0 ]; do
        if [ "$1" = "--" ]; then
            shift
            _shift_c=$((_shift_c + 1))

            break

            fi


        if [ "$1" = "=" ]; then
            echo "[${_defaults_base}.]"

            _base_len=$(expr length "${_defaults_base}.")
            _start_pos=$((_base_len + 1))

            ls "${_defaults_base}."* 2>/dev/null |
                  cut -b ${_start_pos}-

            exit -1
            fi


        if [ "${1:0:1}" = "=" ]; then
            if [ "${1:1:1}" = "/" ]; then
                _defaults_path="${1:1}"
              else
                _defaults_path="${_defaults_base}.${1:1}"
              fi

            shift
            _shift_c=$((_shift_c + 1))

            if [ -f "$_defaults_path" ]; then
                . "$_defaults_path"

              else
                echo "error: defaults file not found: $_defaults_path" >&2
                exit -2

              fi

            _do_default_defaults="False"

            continue
            fi

        if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
            _do_help="True"

            break

            fi

        if [ "${1:0:2}" = "--" ]; then
            _suffix="${1:2}"
            shift


            _suffix_found="False"

            for _suffix_search in $_suffixes; do
                if [ "$_suffix_search" = "$_suffix" ]; then
                    _suffix_found="True"
                    break
                    fi

                done

            if [ "$_suffix_found" = "False" ]; then
                echo "error: suffix \"${_suffix}\" unknown" >&2
                exit -3
                fi


            if [ $# -gt 0 ]; then
                _val="$1"
                shift
                _shift_c=$((_shift_c + 2))

                _varname="${_prefix}_${_suffix}"

                eval "export ${_varname}=\"\$_val\""

              else
                echo "error: suffix \"${_suffix}\" requires a value" >&2
                exit -4

              fi

            continue
            fi

        break
        done


    if [ "$_do_default_defaults" = "True" ]; then
        if [ -f "$_defaults_path_default" ]; then
            . "$_defaults_path_default"

          else
            echo "error: default defaults file not found: $_defaults_path_default" >&2
            exit -5

          fi
        fi


    for _suffix in $_suffixes; do
        _varname="${_prefix}_${_suffix}"
        _default_varname="${_prefix}_${_suffix}_default"
        
        eval "_current_val=\"\${${_varname}}\""
        
        if [ -z "$_current_val" ]; then
            eval "_default_val=\"\${${_default_varname}}\""

            if [ -n "$_default_val" ]; then
                eval "export ${_varname}=\"\$_default_val\""
                fi

            fi

        done


    export envarg_parse_shift_count="$_shift_c"


    if [ "$_do_help" = "True" ]; then
        echo "usage: ${_bin_name} [ -- | = | [ =defaults_file | --var value | -v value ]...] [--help | -h]"

        echo "env:"
        eval "echo \"  ${_prefix}_defaults_path=\\\"\$${_prefix}_defaults_path\\\"\""
        echo "  "

        for _suffix in $_suffixes; do
            _varname="${_prefix}_${_suffix}"
            eval "echo \"  ${_varname}=\\\"\$${_varname}\\\"\""
            done

        exit 0
        fi

    }

