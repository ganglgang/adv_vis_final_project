
function init_layout(){
    $("#grid").html("");
    var total_picture = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','pm','stator_yoke','stator_tooth','stator_winding'];

    var svg = d3.select("#grid").append("svg")
        .attr('width',630)
        .attr('height',630);


    var myimage = svg.selectAll("image").data(total_picture);
    myimage.enter()
        .append('image')
        .attr('xlink:href', function(d){
            return 'images/'+d+'.png'
        })
        .attr('width', 200)
        .attr('height',150)
        .attr('x',function(d,i){
            return -20+200*(i%3)
        })
        .attr('y',function(d,i){
            if (i<3){
                return 0;
            }else if(i>=3 && i<6){
                return 150;
            }else if(i>=6 && i < 9){
                return 300
            }else{
                return 450
            }
        })

}

function features_analysis(name){
    $("#grid").html("");
    $("#legend").html("");

    if(name=='Features Analysis') {

        var total_picture = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','pm','stator_yoke','stator_tooth','stator_winding'];

        var svg = d3.select("#grid").append("svg")
            .attr('width',630)
            .attr('height',630);
        var myimage = svg.selectAll("image").data(total_picture);

            myimage.enter()
                .append('image')
                .attr('xlink:href', function(d){
                    return 'images/'+d+'.png'
                })
                .attr('width', 200)
                .attr('height',150)
                .attr('x',function(d,i){
                    return -20+200*(i%3)
                })
                .attr('y',function(d,i){
                    if (i<3){
                        return 0;
                    }else if(i>=3 && i<6){
                        return 150;
                    }else if(i>=6 && i < 9){
                        return 300
                    }else{
                        return 450
                    }
                })



    }

    if(name === 'Correlation Matrix'){


        d3.select("body").append("div").attr("class", "tip").style("display", "none");

        var data_new = [];

        var data;
        var cols = ['ambient','coolant','u_d','u_q','motor_speed','torque','i_d','i_q','pm','stator_yoke','stator_tooth','stator_winding'];

        var cols_name = ['a','b','c','d','e','f','h','i','j','k','l','m']
        d3.json("data_correlation.json", function(d) {
            data_new = d;


            data =data_new;
            var corr = jz.arr.correlationMatrix(data, cols);
            var extent = d3.extent(corr.map(function(d){ return d.correlation; }).filter(function(d){ return d !== 1; }));

            var grid = data2grid.grid(corr);
            var rows = d3.max(grid, function(d){ return d.row; });

            var margin = {top: 20, bottom: 1, left: 20, right: 1};

            var dim = d3.min([window.innerWidth * .9, window.innerHeight * .93]);
            dim = dim/1.2
            var width = dim - margin.left - margin.right, height = dim - margin.top - margin.bottom;



            var svg = d3.select("#grid").append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", "translate(" + margin.left + ", " + margin.top + ")");

            var padding = .1;

            var x = d3.scaleBand()
                .range([0, width])
                .paddingInner(padding)
                .domain(d3.range(1, rows + 1));

            var y = d3.scaleBand()
                .range([0, height])
                .paddingInner(padding)
                .domain(d3.range(1, rows + 1));

            var c = chroma.scale(["tomato", "white", "steelblue"])
                .domain([extent[0], 0, extent[1]]);

            var x_axis = d3.axisTop(y).tickFormat(function(d, i){ return cols_name[i]; });
            var y_axis = d3.axisLeft(x).tickFormat(function(d, i){ return cols_name[i]; });

            svg.append("g")
                .attr("class", "x axis")
                .call(x_axis);

            svg.append("g")
                .attr("class", "y axis")
                .call(y_axis);

            svg.selectAll("rect")
                .data(grid, function(d){ return d.column_a + d.column_b; })
                .enter().append("rect")
                .attr("x", function(d){ return x(d.column); })
                .attr("y", function(d){ return y(d.row); })
                .attr("width", x.bandwidth())
                .attr("height", y.bandwidth())
                .style("fill", function(d){ return c(d.correlation); })
                .style("opacity", 1e-6)
                .transition()
                .style("opacity", 1);

            svg.selectAll("rect");

            d3.selectAll("rect")
                .on("mouseover", function(d){

                    d3.select(this).classed("selected", true);

                    d3.select(".tip")
                        .style("display", "block")
                        .html(d.column_x + ", " + d.column_y + ": " + d.correlation.toFixed(2));

                    var row_pos = y(d.row);
                    var col_pos = x(d.column);
                    var tip_pos = d3.select(".tip").node().getBoundingClientRect();
                    var tip_width = tip_pos.width;
                    var tip_height = tip_pos.height;
                    var grid_pos = d3.select("#grid").node().getBoundingClientRect();
                    var grid_left = grid_pos.left;
                    var grid_top = grid_pos.top;

                    var left = grid_left + col_pos + margin.left + (x.bandwidth() / 2) - (tip_width / 2);
                    var top = grid_top + row_pos + margin.top - tip_height - 5;

                    d3.select(".tip")
                        .style("left", left + "px")
                        .style("top", top + "px");

                    d3.select(".x.axis .tick:nth-of-type(" + d.column + ") text").classed("selected", true);
                    d3.select(".y.axis .tick:nth-of-type(" + d.row + ") text").classed("selected", true);
                    d3.select(".x.axis .tick:nth-of-type(" + d.column + ") line").classed("selected", true);
                    d3.select(".y.axis .tick:nth-of-type(" + d.row + ") line").classed("selected", true);

                })
                .on("mouseout", function(){
                    d3.selectAll("rect").classed("selected", false);
                    d3.select(".tip").style("display", "none");
                    d3.selectAll(".axis .tick text").classed("selected", false);
                    d3.selectAll(".axis .tick line").classed("selected", false);
                });

            // legend scale
            var legend_top = 15;
            var legend_height = 15;

            var legend_svg = d3.select("#legend").append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", legend_height + legend_top)
                .append("g")
                .attr("transform", "translate(" + margin.left + ", " + legend_top + ")");

            var defs = legend_svg.append("defs");

            var gradient = defs.append("linearGradient")
                .attr("id", "linear-gradient");

            var stops = [{offset: 0, color: "tomato", value: extent[0]}, {offset: .5, color: "white", value: 0}, {offset: 1, color: "steelblue", value: extent[1]}];

            gradient.selectAll("stop")
                .data(stops)
                .enter().append("stop")
                .attr("offset", function(d){ return (100 * d.offset) + "%"; })
                .attr("stop-color", function(d){ return d.color; });

            legend_svg.append("rect")
                .attr("width", width)
                .attr("height", legend_height)
                .style("fill", "url(#linear-gradient)");

            legend_svg.selectAll("text")
                .data(stops)
                .enter().append("text")
                .attr("x", function(d){ return width * d.offset; })
                .attr("dy", -3)
                .style("text-anchor", function(d, i){ return i == 0 ? "start" : i == 1 ? "middle" : "end"; })
                .text(function(d, i){ return d.value.toFixed(2) + (i == 2 ? ">" : ""); })
        });
    }
    if (name === 'PCA'){
        console.log('hahah')

    }

}