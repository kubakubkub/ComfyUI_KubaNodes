const data = { count: 6, groups: [["windows", 3], ["columns", 2], ["my group", 1]],
  rows: [{ label: "floors", tags: [["floor_1", 3], ["floor_2", 3]] }, { label: "sections", tags: [["left", 2], ["centre", 4]] }],
  regions: [["windows_01", "windows", ["floor_1", "left"]], ["windows_02", "windows", ["floor_1", "centre"]], ["windows_03", "windows", ["floor_2", "centre"]],
    ["columns_01", "columns", ["floor_1", "left"]], ["columns_02", "columns", ["floor_2", "centre"]], ["the door", "my group", ["floor_2", "centre"]]] };
let fails = 0;
const check = (name, ok, d) => { console.log((ok ? "ok   " : "FAIL ") + name + (ok ? "" : "  " + JSON.stringify(d))); if (!ok) fails++; };
const p = { groups: ["windows", "columns"], rows: { floors: ["floor_1"], sections: [] }, names: [] };
check("text: any of the groups, on the floor", textOf(p) === "group:windows tag:floor_1, group:columns tag:floor_1", textOf(p));
check("count", chosen(p, data).map(r => r[0]).join() === "windows_01,windows_02,columns_01", chosen(p, data));
check("round trip", JSON.stringify(pickOf(textOf(p), data).groups) === JSON.stringify(p.groups) && pickOf(textOf(p), data).rows.floors[0] === "floor_1");
check("order of terms is free, capitals too", pickOf("TAG:floor_1 Group:Windows", data)?.groups[0] === "windows");
check("empty and * = nothing chosen", termsOf(pickOf("", data)).length === 0 && termsOf(pickOf("*", data)).length === 0);
check("hand typed stays hand typed", pickOf("W_F1_*", data) === null && pickOf("group:windows !tag:left", data) === null && pickOf("group:nothing", data) === null);
check("a partial cross product is hand typed", pickOf("group:windows tag:floor_1, group:columns tag:floor_2", data) === null);
const q = { groups: ["my group"], rows: {}, names: ["the door", "windows_03"] };
check("a space in a name does not split the term", textOf(q) === "group:my?group, name:the?door, windows_03", textOf(q));
check("... and reads back", pickOf(textOf(q), data)?.names.join() === "the door,windows_03" && pickOf(textOf(q), data).groups[0] === "my group");
check("names add to the rows", chosen({ groups: ["columns"], rows: { floors: ["floor_2"] }, names: ["windows_01"] }, data).map(r => r[0]).join() === "windows_01,columns_02");
check("reserved words and '//' in names are written so the rules read them as names",
  textOf({ groups: ["a//b"], rows: {}, names: ["then", "plan", "ok"] }) === "group:a/?b, name:then, name:plan, ok", textOf({ groups: ["a//b"], rows: {}, names: ["then", "plan", "ok"] }));
check("only tags", textOf({ groups: [], rows: { floors: ["floor_1", "floor_2"], sections: ["left"] }, names: [] }) === "tag:floor_1 tag:left, tag:floor_2 tag:left");
process.exit(fails ? 1 : 0);
